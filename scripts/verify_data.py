"""
Gate before training: rebuilds the corpus and checks it is big, clean, and balanced.
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
single_turns = [r for r in rows if "q" in r]
multi_turns = [r for r in rows if "messages" in r]
answers = {r["a"] for r in single_turns}

# Check refusal count
refusal_prefixes = ("that one isn't in my marvel", "sorry, that's outside", "not my department",
                    "i'm all marvel", "that's beyond my marvel", "hmm, that doesn't belong",
                    "my archives cover marvel", "error 404: marvel", "i am programmed specifically",
                    "that falls outside stark", "i'm afraid that lies", "that's not part of the marvel",
                    "i specialize only", "my database focuses", "i can't calculate that", "stark industries ai")
refusal_count = sum(1 for r in single_turns if any(r["a"].lower().startswith(p) for p in refusal_prefixes))
refusal_pct = refusal_count / max(1, len(rows)) * 100

problems = []
if est_tokens < a.min_tokens:
    problems.append(f"pretrain text too small: ~{est_tokens/1e6:.1f}M tokens < {a.min_tokens/1e6:.0f}M")
if len(rows) < a.min_pairs:
    problems.append(f"too few SFT rows: {len(rows)} < {a.min_pairs}")
if docs < 500:
    problems.append(f"too few documents: {docs}")
if refusal_pct > 1.5:
    problems.append(f"refusal attractor detected: {refusal_count} refusals ({refusal_pct:.2f}% > 1.5% ceiling)")

print(f"VERIFY pretrain: ~{est_tokens/1e6:.1f}M tokens across {docs} docs")
print(f"VERIFY sft: {len(rows)} total rows (single-turn: {len(single_turns)}, multi-turn: {len(multi_turns)}, refusals: {refusal_count} [{refusal_pct:.2f}%])")
print("VERIFY " + ("OK: Corpus is balanced and ready for training!" if not problems else "NOT READY: " + "; ".join(problems)))
sys.exit(1 if problems else 0)
