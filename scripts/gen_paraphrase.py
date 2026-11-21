"""
Ask Gemini (agy) to REWORD each training question, so the model answers correctly however a person phrases it.
  python scripts/gen_paraphrase.py --workers 10        (resumable)
Reads data/para_src.jsonl (frozen real Q&A). Writes data/raw/para/*.jsonl: {"i": row, "k": 0|1|2, "q": reworded question}
k=0,1 go into training; k=2 is a held-out 'exam' (scripts/eval_memory.py) to measure real robustness.
"""
import argparse, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

MODEL = "gemini-3.8-flash-medium"
BATCH = 80
PROMPT = """Below are numbered questions. For EACH one write 3 different rewordings that ask the SAME thing (same meaning, different words).
Vary the style: one casual or with a typo, one formal, one short or keyword-like. Never change the meaning or add new facts.
Output ONLY JSON Lines, no markdown fences, one object per rewording: {{"i": <number>, "q": "<reworded question>"}}
Questions:
{items}"""


def job(start, rows):
    path = f"data/raw/para/{start:06d}.jsonl"
    if os.path.exists(path):
        return path, "skip"
    chunk = rows[start:start + BATCH]
    items = "\n".join(f"{start + j}. {r['q']}" for j, r in enumerate(chunk))
    try:
        out = subprocess.run(["agy", "--print", PROMPT.format(items=items), "--model", MODEL, "--output-format", "text"],
                             capture_output=True, text=True, encoding="utf-8", timeout=900).stdout
        seen, lines = {}, []
        for line in out.splitlines():
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                d = json.loads(m.group(0)); i = int(d["i"]); q = d["q"].strip()
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if start <= i < start + len(chunk) and 3 <= len(q) <= 300:
                k = seen.get(i, 0); seen[i] = k + 1
                if k < 3:
                    lines.append(json.dumps({"i": i, "k": k, "q": q}, ensure_ascii=False))
        if len(lines) < len(chunk):
            return path, f"too few ({len(lines)})"
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        return path, f"{len(lines)} rewordings"
    except Exception as e:
        return path, f"error {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    os.makedirs("data/raw/para", exist_ok=True)
    rows = [json.loads(l) for l in open("data/para_src.jsonl", encoding="utf-8")]
    with ThreadPoolExecutor(a.workers) as ex:
        for path, status in ex.map(lambda s: job(s, rows), range(0, len(rows), BATCH)):
            print(path, status, flush=True)
