"""
Generate training data with the `agy` CLI (Gemini). Re-run as often as you like; each (topic, angle) goes to its own file
and is skipped if it already exists, so interrupted runs resume.

  python scripts/gen_data.py --kind qa    --workers 4           # Q&A for the chat stage
  python scripts/gen_data.py --kind prose --workers 4           # articles for pretraining
  python scripts/gen_data.py --kind qa --limit 3 --workers 1    # quick test
"""
import argparse, hashlib, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

MODEL = "gemini-3.8-flash-medium"
TOPICS = [t.strip() for t in open(os.path.join(os.path.dirname(__file__), "topics.txt"), encoding="utf-8") if t.strip()]

# Different "angles" make Gemini produce different questions for the same topic => more variety.
QA_ANGLES = [
    "beginner-friendly questions a new fan would ask",
    "detailed trivia questions for hardcore fans",
    "comic-book origins and history",
    "MCU movie and TV facts (actors, plot, release years)",
    "who/what/when/where/why/how mix with casual chatty phrasing (lowercase, slang, typos allowed in the question)",
    "comparison and 'who would win' style questions, answered with grounded reasoning",
    "short one-line answers",
    "multi-sentence explanations (2-3 sentences)",
    "questions about specific storylines and what happened in them",
    "questions about actors, directors and creators",
    "questions about powers, weaknesses and abilities",
    "questions about relationships, allies and enemies",
    "questions about locations, items and technology",
    "yes/no and true/false style questions, answered with a short explanation",
    "questions a kid would ask, with simple answers",
    "'explain like I'm five' and 'tell me about' style requests",
]
PROSE_ANGLES = ["comic-book history", "MCU appearances", "powers, abilities and weaknesses",
                "relationships and key storylines", "cultural impact and trivia", "origin story told as a narrative",
                "the character's most famous comic arcs", "behind the scenes: creators, actors and production",
                "allies, enemies and rivalries", "how it connects to the wider Marvel universe",
                "a fan guide for newcomers", "timeline of key events", "notable quotes and iconic moments",
                "differences between the comics and the movies", "lesser-known facts"]

QA = """Generate 80 diverse question-answer pairs about the Marvel topic: {topic}.
Focus: {angle}.
Rules: answers are 1-3 sentences, factual, friendly, written as F.R.I.D.A.Y., a helpful Marvel assistant.
No duplicates. If unsure of a fact, skip that question. Only released/announced facts, never invent plot details.
Output ONLY JSON Lines, no markdown fences, one object per line: {{"q": "...", "a": "..."}}"""

PROSE = """Write a ~700 word encyclopedic, factual article about: {topic} (Marvel). Angle: {angle}.
Plain text only: no markdown, no headings, no lists, no preamble. Only facts you are sure about."""


def run_agy(prompt: str) -> str:
    r = subprocess.run(["agy", "--print", prompt, "--model", MODEL, "--output-format", "text"],
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    return r.stdout


def job(args):
    kind, topic, angle = args
    tag = hashlib.md5(f"{kind}|{topic}|{angle}".encode()).hexdigest()[:10]
    folder = "data/raw/sft" if kind == "qa" else "data/raw/pretrain"
    path = f"{folder}/{tag}.{'jsonl' if kind == 'qa' else 'txt'}"
    if os.path.exists(path):
        return path, "skip"
    try:
        out = run_agy((QA if kind == "qa" else PROSE).format(topic=topic, angle=angle))
        if kind == "qa":
            rows = []
            for line in out.splitlines():
                m = re.search(r"\{.*\}", line)
                if not m:
                    continue
                try:
                    d = json.loads(m.group(0))
                    if d.get("q") and d.get("a"):
                        rows.append(json.dumps({"q": d["q"].strip(), "a": d["a"].strip()}, ensure_ascii=False))
                except json.JSONDecodeError:
                    pass
            if not rows:
                return path, "no rows"
            open(path, "w", encoding="utf-8").write("\n".join(rows) + "\n")
            return path, f"{len(rows)} pairs"
        text = " ".join(out.split())
        if len(text) < 800:
            return path, "too short"
        open(path, "w", encoding="utf-8").write(text + "\n")
        return path, f"{len(text)} chars"
    except Exception as e:                       # network hiccup etc: just retry on next run
        return path, f"error {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["qa", "prose"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs("data/raw/sft", exist_ok=True); os.makedirs("data/raw/pretrain", exist_ok=True)
    angles = QA_ANGLES if a.kind == "qa" else PROSE_ANGLES
    jobs = [(a.kind, t, g) for g in angles for t in TOPICS][: a.limit]
    with ThreadPoolExecutor(a.workers) as ex:
        for path, status in ex.map(job, jobs):
            print(path, status, flush=True)
