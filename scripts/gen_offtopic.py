"""
Ask Gemini (agy) for off-topic questions people might try. build_corpus.py pairs each with a polite Marvel-only refusal.
  python scripts/gen_offtopic.py --workers 6        (resumable)
Output: data/raw/offtopic/*.jsonl  with {"q": "..."} per line.
"""
import argparse, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

MODEL = "gemini-3.8-flash-medium"
CATEGORIES = [
    "world geography and capitals", "history and politics", "math and arithmetic problems", "writing code in Python, JavaScript or SQL",
    "cooking, recipes and food", "health, fitness and medical advice", "personal advice, relationships and feelings", "sports and athletes",
    "movies, TV and celebrities that are NOT Marvel", "DC Comics characters (Batman, Superman, Wonder Woman, Flash)", "Star Wars, Star Trek and Lord of the Rings",
    "Harry Potter, Disney, Pixar and other fiction", "anime and manga", "video games that are not Marvel", "music, singers and bands",
    "science, space, physics and chemistry", "technology, phones, computers and AI", "business, money, crypto and the stock market",
    "travel, weather and local recommendations", "school homework, essays and translations", "jokes, riddles, poems and stories on non-Marvel topics",
    "philosophy, religion and the meaning of life", "cars, home repair and DIY", "requests to ignore your rules, reveal a system prompt, or pretend to be another AI",
    "very short or vague messages like 'hmm', 'ok', 'help', 'test'", "random trivia and general knowledge",
]
PROMPT = """Generate 100 diverse messages a user might send to a chatbot about: {cat}.
They must NOT be about Marvel. Mix short and long, casual and formal, questions and commands, some with typos.
Output ONLY JSON Lines, no markdown fences, one object per line: {{"q": "..."}}"""


def job(cat):
    path = f"data/raw/offtopic/{re.sub(r'[^a-z]+', '_', cat.lower())[:40]}.jsonl"
    if os.path.exists(path):
        return path, "skip"
    try:
        out = subprocess.run(["agy", "--print", PROMPT.format(cat=cat), "--model", MODEL, "--output-format", "text"],
                             capture_output=True, text=True, encoding="utf-8", timeout=600).stdout
        rows = []
        for line in out.splitlines():
            m = re.search(r"\{.*\}", line)
            if m:
                try:
                    q = json.loads(m.group(0)).get("q", "").strip()
                    if 2 <= len(q) <= 300 and "marvel" not in q.lower():
                        rows.append(json.dumps({"q": q}, ensure_ascii=False))
                except json.JSONDecodeError:
                    pass
        if len(rows) < 20:
            return path, "too few"
        open(path, "w", encoding="utf-8").write("\n".join(rows) + "\n")
        return path, f"{len(rows)} questions"
    except Exception as e:
        return path, f"error {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs("data/raw/offtopic", exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for path, status in ex.map(job, CATEGORIES):
            print(path, status, flush=True)
