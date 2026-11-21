"""
Training loop for both stages.
  python -m src.training.train --stage pretrain            (learn language + Marvel facts from raw text)
  python -m src.training.train --stage sft --init checkpoints/pretrain_best.pt   (learn to answer questions)
Add --resume to continue an interrupted run. Add --max_steps N for a quick smoke test.

One training step =  1) get a batch  2) forward pass -> loss  3) backward pass -> gradients
                     4) clip gradients  5) optimizer updates weights
"""
import argparse, json, os, random, time
import numpy as np
import torch
from src.utils.config import load_config
from src.core.functional import cross_entropy
from src.models.friday import Friday
from src.tokenization.tokenizer import Tok
from src.training.optim import AdamW, lr_at
from src.inference.generate import generate


# ---------------------------------------------------------------- data
def load_pretrain_tokens(tok, path="data/pretrain.txt", cache="data/pretrain.bin"):
    """Turn the whole text file into ONE long array of token ids:  <s> doc1 </s> <s> doc2 </s> ..."""
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return np.fromfile(cache, dtype=np.uint16)
    out = []
    for doc in open(path, encoding="utf-8"):
        doc = doc.strip()
        if doc:
            out += [tok.bos] + tok.encode(doc) + [tok.eos]
    arr = np.array(out, dtype=np.uint16)         # vocab < 65536 so 2 bytes per token is enough
    arr.tofile(cache)
    return arr


def pretrain_batch(data, B, T, device):
    """Random windows. x = tokens[i : i+T], y = tokens[i+1 : i+T+1] (the NEXT token at every position)."""
    ix = np.random.randint(0, len(data) - T - 1, size=B)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def load_sft(tok, T, path="data/sft.jsonl"):
    ex = []
    for row in open(path, encoding="utf-8"):
        r = json.loads(row)
        if "messages" in r:
            ids, labels = tok.chat_multiturn(r["messages"])
        else:
            ids, labels = tok.chat_example(r["q"], r["a"])
        ex.append((ids[:T + 1], labels[:T + 1]))
    return ex


def sft_batch(examples, device):
    """Pad a list of (ids, labels) to the same length. Padding label = -100 so it is ignored."""
    L = max(len(i) for i, _ in examples)
    x = torch.zeros(len(examples), L - 1, dtype=torch.long)
    y = torch.full((len(examples), L - 1), -100, dtype=torch.long)
    for r, (ids, lab) in enumerate(examples):
        x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
        y[r, :len(ids) - 1] = torch.tensor(lab[1:])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- helpers
def loss_fn(model, x, y):
    logits = model(x)                                              # (B, T, V)
    return cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def evaluate(model, get_batch, n=20):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = get_batch()
        losses.append(loss_fn(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)


def make_optimizer(model, lr, wd):
    """Weight decay only on 2-D matrices (not on norm scales)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    return AdamW([{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}], lr=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--init", default=None, help="start from these weights (used for sft)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs from the config")
    ap.add_argument("--final", action="store_true", help="sft: memorise the data (train on nearly all pairs, keep the LAST model, not the best-on-held-out)")
    a = ap.parse_args()

    c = load_config(a.config)
    dev = "cuda"
    torch.manual_seed(1337); np.random.seed(1337); random.seed(1337)
    tok = Tok()
    assert tok.vocab_size == c["vocab_size"], f"tokenizer has {tok.vocab_size} pieces, config says {c['vocab_size']}"
    T, B, G = c["context_length"], c["batch_size"], c["grad_accum"]

    model = Friday(c).to(dev)
    print(f"model parameters: {model.num_params()/1e6:.2f}M")
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev)["model"])
        print("loaded", a.init)

    # ---- data + schedule
    if a.stage == "pretrain":
        data = load_pretrain_tokens(tok)
        n_val = max(T * 20, len(data) // 100)
        train_d, val_d = data[:-n_val], data[-n_val:]
        steps_per_epoch = max(1, len(train_d) // (B * G * T))
        get_train = lambda: pretrain_batch(train_d, B, T, dev)
        get_val = lambda: pretrain_batch(val_d, B, T, dev)
        epochs, peak = c["pretrain_epochs"], c["pretrain_lr"]
        probe = [tok.bos] + tok.encode("Iron Man is")
        print(f"tokens: {len(data)/1e6:.2f}M | steps/epoch: {steps_per_epoch}")
    else:
        ex = load_sft(tok, T)
        random.shuffle(ex)
        n_val = 100 if a.final else max(1, len(ex) // 50)
        val_ex, train_ex = ex[:n_val], ex[n_val:]
        steps_per_epoch = max(1, len(train_ex) // (B * G))
        order = {"i": 0}

        def get_train():
            if order["i"] + B > len(train_ex):
                random.shuffle(train_ex); order["i"] = 0
            b = train_ex[order["i"]:order["i"] + B]; order["i"] += B
            return sft_batch(b, dev)
        get_val = lambda: sft_batch(random.sample(val_ex, min(B, len(val_ex))), dev)
        epochs, peak = a.epochs or c["sft_epochs"], c["sft_lr"]
        probe = None
        print(f"examples: {len(train_ex)} | steps/epoch: {steps_per_epoch}")

    total = a.max_steps or steps_per_epoch * epochs
    opt = make_optimizer(model, peak, c["weight_decay"])
    # NOTE: plain fp32 on purpose. The GTX 1650 Ti has no tensor cores and fp16 measured 4x SLOWER than fp32.
    step, best = 0, float("inf")
    last_path = f"checkpoints/{a.stage}_last.pt"
    os.makedirs("checkpoints", exist_ok=True)
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print("resumed at step", step)

    model.train()
    t0 = time.time()
    while step < total:
        lr = lr_at(step, total, peak, c["warmup_steps"], c["min_lr_ratio"])
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(G):                                           # gradient accumulation
            x, y = get_train()
            loss = loss_fn(model, x, y) / G                          # forward pass
            loss.backward()                                          # BACKPROP: fills p.grad
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])   # (library helper: just rescales grads)
        opt.step()                                                   # AdamW update
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 20 == 0:
            print(f"step {step}/{total} | loss {loss_sum:.4f} | lr {lr:.2e} | {(time.time()-t0)/step:.2f}s/step", flush=True)
        if step % c["eval_every"] == 0 or step == total:
            v = evaluate(model, get_val)
            print(f">>> step {step} VAL loss {v:.4f}", flush=True)
            if probe:
                out = generate(model, probe, max_new=40, ctx=T, eos_id=tok.eos)
                print("    sample:", tok.decode(probe + out))
            model.train()
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "config": c}, f"checkpoints/{a.stage}_best.pt")
                print(f"    *** Saved new best checkpoint (val loss: {best:.4f})")
        if step % c["save_every"] == 0 or step == total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best": best, "config": c}, last_path)
    print("done. best val loss:", best)


if __name__ == "__main__":
    main()
