"""Text generation: the model predicts ONE next token at a time, we pick one, repeat."""
import torch
from src.core.functional import softmax


@torch.no_grad()
def generate(model, ids, max_new=200, temperature=0.3, top_k=30, top_p=0.85, rep_penalty=1.1,
             eos_id=None, ctx=512, device="cuda"):
    """
    In : ids  list[int] prompt token ids
    Out: list[int] only the newly generated ids
    temperature: <1 = careful/boring, >1 = wild.  top_k/top_p: only choose among likely tokens.
    rep_penalty: makes already-used tokens a bit less likely (stops loops).
    """
    model.eval()
    seq = list(ids)
    new = []
    for _ in range(max_new):
        x = torch.tensor([seq[-ctx:]], device=device)
        logits = model(x)[0, -1].float()                               # (vocab,) scores for the next token
        for t in set(seq[-64:]):                                       # 1) punish repeats
            logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
        logits = logits / max(temperature, 1e-5)                       # 2) temperature
        if top_k:                                                      # 3) keep only the k best
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = float("-inf")
        probs = softmax(logits, dim=-1)
        if top_p < 1.0:                                                # 4) keep smallest set with prob >= p
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=0)
            drop = cum - sp > top_p                                    # tokens after we already reached p
            sp[drop] = 0
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
            probs = probs / probs.sum()
        nxt = int(torch.multinomial(probs, 1))                         # 5) roll the dice
        if nxt == eos_id:
            break
        seq.append(nxt); new.append(nxt)
    return new
