"""
Two scores for a trained model (greedy decoding, no randomness):
  1) TRAINED questions: how many does it still answer correctly?
  2) REWORDED questions it never saw (held-out paraphrases, k=2): does it still answer correctly?
  python scripts/eval_memory.py [--n 200] [--ckpt checkpoints/sft_best.pt]
"EXACT" = identical answer, "CLOSE" = word-overlap F1 >= 0.8.
"""
import argparse, collections, glob, json, os, random, re, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.friday import Friday
from src.tokenization.tokenizer import Tok
from src.inference.generate import generate

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--ckpt", default="checkpoints/sft_best.pt")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cuda")
model = Friday(ck["config"]).cuda(); model.load_state_dict(ck["model"])
tok = Tok()
norm = lambda s: re.sub(r"\W+", " ", s.lower()).split()


def score(rows, title, show=3):
    exact = close = 0
    for i, r in enumerate(rows):
        out = generate(model, tok.prompt_ids(r["q"]), max_new=150, temperature=1.0, top_k=1, top_p=1.0,
                       rep_penalty=1.0, ctx=ck["config"]["context_length"], eos_id=tok.eos)
        got, want = norm(tok.decode(out)), norm(r["a"])
        f1 = 2 * sum((collections.Counter(got) & collections.Counter(want)).values()) / max(1, len(got) + len(want))
        exact += got == want; close += f1 >= 0.8
        if i < show:
            print(f"Q: {r['q']}\n  WANT: {r['a'][:140]}\n  GOT : {tok.decode(out)[:140]}")
    print(f"{title}: EXACT {exact/len(rows):.0%} | CLOSE {close/len(rows):.0%} (n={len(rows)})\n")


rows, seen = [], set()
for line in open("data/sft.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["q"] not in seen:
        seen.add(r["q"]); rows.append(r)
random.Random(1).shuffle(rows)
score(rows[: a.n], "TRAINED questions")

if os.path.exists("data/para_src.jsonl"):
    base = [json.loads(l) for l in open("data/para_src.jsonl", encoding="utf-8")]
    exam = []
    for f in sorted(glob.glob("data/raw/para/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if d.get("k") == 2:
                exam.append({"q": d["q"], "a": base[d["i"]]["a"]})
    random.Random(2).shuffle(exam)
    if exam:
        score(exam[: a.n], "UNSEEN REWORDED questions")
