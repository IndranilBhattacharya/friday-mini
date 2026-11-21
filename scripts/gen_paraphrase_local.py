"""
Reword training questions with a LOCAL Ollama model (no Gemini, no internet). Safe task: rewording changes the
wording, not the facts. Every rewording is checked with the nomic-embed-text model and dropped if its meaning drifted.

  python scripts/gen_paraphrase_local.py --workers 2                  # GPU must be FREE (training finished)
  python scripts/gen_paraphrase_local.py --cpu --limit 1              # tiny CPU test, safe while training runs
Skips questions that already have Gemini rewordings. Resumable. Output: data/raw/para/L*.jsonl, same format as Gemini's.
"""
import argparse, glob, json, os, re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

OLLAMA = "http://localhost:11434"
BATCH = 10
MIN_SIM = 0.80          # cosine similarity between original and reworded question (meaning must stay the same)
PROMPT = """Reword each numbered question so it asks exactly the SAME thing in different words.
Keep every name, title and detail. Do not answer the questions. Do not add facts.
Vary the style (casual, formal or short). Reply with ONLY the numbered rewordings, one per line, like:
1. reworded question
2. reworded question

Questions:
{items}"""


def embed(texts, opts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": "nomic-embed-text", "input": texts, "options": opts}, timeout=300)
    return np.array(r.json()["embeddings"])


def job(start, rows, covered, a):
    path = f"data/raw/para/L{start:06d}.jsonl"
    idx = [i for i in range(start, min(start + BATCH, len(rows))) if i not in covered]
    if os.path.exists(path) or not idx:
        return path, "skip"
    opts = {"num_gpu": 0, "num_thread": 2} if a.cpu else {}
    try:
        items = "\n".join(f"{n + 1}. {rows[i]['q']}" for n, i in enumerate(idx))
        r = requests.post(f"{OLLAMA}/api/generate", json={"model": a.model, "prompt": PROMPT.format(items=items), "stream": False,
                          "options": {"temperature": 0.7, **opts}}, timeout=900).json()["response"]
        got = {}
        for line in r.splitlines():
            m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
            if m and 1 <= int(m.group(1)) <= len(idx):
                got[idx[int(m.group(1)) - 1]] = m.group(2).strip().strip('"')
        keep = [(i, q) for i, q in got.items() if 3 <= len(q) <= 300 and q.lower() != rows[i]["q"].lower()]
        out = []
        if keep:
            e = embed([rows[i]["q"] for i, _ in keep] + [q for _, q in keep], opts)
            n = len(keep)
            for j, (i, q) in enumerate(keep):
                sim = float(e[j] @ e[n + j] / (np.linalg.norm(e[j]) * np.linalg.norm(e[n + j])))
                if sim >= MIN_SIM:
                    out.append(json.dumps({"i": i, "k": 0, "q": q}, ensure_ascii=False))
        open(path, "w", encoding="utf-8").write("\n".join(out) + ("\n" if out else ""))
        return path, f"{len(out)}/{len(idx)} kept"
    except Exception as e:
        return path, f"error {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true", help="run on CPU only (safe while GPU training is running)")
    ap.add_argument("--limit", type=int, default=None, help="only first N batches (test)")
    a = ap.parse_args()
    os.makedirs("data/raw/para", exist_ok=True)
    rows = [json.loads(l) for l in open("data/para_src.jsonl", encoding="utf-8")]
    covered = set()
    for f in glob.glob("data/raw/para/0*.jsonl"):            # Gemini files are named 000000.jsonl etc.
        for line in open(f, encoding="utf-8"):
            covered.add(json.loads(line)["i"])
    starts = list(range(0, len(rows), BATCH))[: a.limit] if a.limit is None else \
        [s for s in range(0, len(rows), BATCH) if any(i not in covered for i in range(s, min(s + BATCH, len(rows))))][: a.limit]
    with ThreadPoolExecutor(a.workers) as ex:
        for path, status in ex.map(lambda s: job(s, rows, covered, a), starts):
            print(path, status, flush=True)
