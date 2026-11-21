"""
Gate before training: rebuilds the corpus and checks it is big and clean enough.
Exit code 0 = ready, 1 = not ready.   python scripts/verify_data.py [--min_tokens 18e6] [--min_pairs 20000]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--min_tokens", type=float, default=18e6)
ap.add_argument("--min_pairs", type=int, default=20000)
a = ap.parse_args()

build_corpus.main()                                   # clean + dedupe into data/pretrain.txt and data/sft.jsonl
size = os.path.getsize("data/pretrain.txt")
est_tokens = size / 4.3                               # ~4.3 characters per token for English with an 8k vocab
docs = sum(1 for _ in open("data/pretrain.txt", encoding="utf-8"))
rows = [json.loads(l) for l in open("data/sft.jsonl", encoding="utf-8")]
answers = {r["a"] for r in rows}
problems = []
if est_tokens < a.min_tokens:
    problems.append(f"pretrain text too small: ~{est_tokens/1e6:.1f}M tokens < {a.min_tokens/1e6:.0f}M")
if len(rows) < a.min_pairs:
    problems.append(f"too few Q&A pairs: {len(rows)} < {a.min_pairs}")
if len(answers) < 0.8 * len(rows):
    problems.append(f"too many repeated answers: {len(answers)} unique of {len(rows)}")
if docs < 500:
    problems.append(f"too few documents: {docs}")
print(f"VERIFY pretrain ~{est_tokens/1e6:.1f}M tokens, {docs} docs | sft {len(rows)} pairs ({len(answers)} unique answers)")
print("VERIFY " + ("OK" if not problems else "NOT READY: " + "; ".join(problems)))
sys.exit(1 if problems else 0)
