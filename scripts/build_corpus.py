"""
Clean + merge raw data into the two training files.
  data/raw/pretrain/*.txt (one doc per line) -> data/pretrain.txt
  data/raw/sft/*.jsonl                       -> data/sft.jsonl (deduplicated)
Run: python scripts/build_corpus.py
"""
import filecmp, glob, json, os, re, shutil

GREET = ["Hi", "Hello", "Hey", "Hey there", "Good morning", "Good evening", "Yo", "Hi Friday", "Hello Friday",
         "Hey Friday", "Howdy", "Greetings", "What's up?", "Hi there!", "Hello!", "Sup", "Good afternoon", "Hey buddy"]
GREET_A = ["Hello! I'm F.R.I.D.A.Y. mini. Ask me anything about the Marvel universe.",
           "Hey there! Ready to talk Marvel? Ask me about heroes, villains, or the MCU.",
           "Hi! I'm F.R.I.D.A.Y. mini, your Marvel assistant. Who or what shall we talk about?"]
ABOUT = [("Who are you?", "I'm F.R.I.D.A.Y. mini, a tiny Marvel-expert language model built from scratch. Ask me anything about Marvel!"),
         ("What are you?", "I'm a small language model trained only on Marvel knowledge, so I'm best at heroes, villains and the MCU."),
         ("What is your name?", "My name is F.R.I.D.A.Y. mini, a small Marvel assistant."),
         ("Who made you?", "I was built from scratch as a learning project, trained on Marvel comics and MCU knowledge."),
         ("What can you do?", "I answer questions about Marvel heroes, villains, comics, movies and the MCU."),
         ("Are you an AI?", "Yes, I'm F.R.I.D.A.Y. mini, a small AI language model that knows Marvel."),
         ("Are you Jarvis?", "No, I'm F.R.I.D.A.Y. mini, a small Marvel assistant. Jarvis is Tony Stark's original AI."),
         ("Thanks!", "You're welcome! Ask me anything else about Marvel."),
         ("Thank you", "Happy to help! Any more Marvel questions?"),
         ("Bye", "Goodbye! Come back any time for more Marvel talk."), ("How are you?", "All systems online! Ready for your Marvel questions.")]
OFF_Q = ["What's the capital of France?", "What is the capital of Japan?", "What's the weather today?", "Write me a poem about cheese.",
         "Write Python code to sort a list.", "How do I bake a chocolate cake?", "What is 2 + 2?", "Who is the president of the United States?",
         "Who won the football world cup?", "Tell me about Harry Potter.", "Who is Taylor Swift?", "Who is Batman?", "Tell me about Superman.",
         "What is the meaning of life?", "How do I lose weight?", "What is the stock price of Apple?", "Translate hello to Spanish.",
         "Who is Naruto?", "What is Pokemon?", "How do I fix my car?", "Explain quantum physics.", "Who invented the telephone?",
         "What is the tallest mountain?", "Recommend me a restaurant.", "What is bitcoin?", "Tell me a joke about cats.",
         "Who is Darth Vader?", "What is Python?", "How do I learn guitar?", "What is the speed of light?"]
OFF_A = ["That one isn't in my Marvel universe, boss. You'd have to check with Tony... though he'd probably send you an invoice!",
         "Sorry, that's outside my Marvel universe. I'd ask Tony, but he's busy building a suit. Want to hear about the Avengers instead?",
         "Not my department! I only know Marvel. For anything else you'll need to check with Tony, and good luck getting a straight answer.",
         "I'm all Marvel, all the time, and that question isn't part of it. Try Tony, or ask me about the X-Men!",
         "That's beyond my Marvel database, boss. Even Doctor Strange couldn't find that one in 14 million futures. Ask me about a hero instead!",
         "Hmm, that doesn't belong to my Marvel universe. Check with Tony for that, and I'll happily take any question about the MCU.",
         "My universe is Marvel only, so I can't help there. Even Thor would say that's above his pay grade. Ask me about Asgard instead!",
         "Error 404: Marvel not found. That's outside my universe, boss. Ask me about heroes, villains or the MCU!"]
IDENTITY = [(q, GREET_A[i % 3]) for i, q in enumerate(GREET)] + ABOUT + [(q, OFF_A[i % 3]) for i, q in enumerate(OFF_Q)]
REPEAT = 12      # these small-talk examples are repeated so the model really learns them


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    docs = 0
    with open("data/pretrain.tmp", "w", encoding="utf-8") as out:
        for f in sorted(glob.glob("data/raw/pretrain/*.txt")):
            for line in open(f, encoding="utf-8"):
                line = clean(line)
                if len(line) > 500:
                    out.write(line + "\n"); docs += 1
    if os.path.exists("data/pretrain.txt") and filecmp.cmp("data/pretrain.tmp", "data/pretrain.txt", shallow=False):
        os.remove("data/pretrain.tmp")                    # unchanged: keep the cached token file valid
    else:
        shutil.move("data/pretrain.tmp", "data/pretrain.txt")
    seen, rows = set(), []
    pairs = [{"q": q, "a": a} for q, a in IDENTITY]
    for f in sorted(glob.glob("data/raw/sft/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    for p in pairs:
        q, a = clean(p.get("q", "")), clean(p.get("a", ""))
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if 3 <= len(q) <= 300 and 5 <= len(a) <= 600 and key not in seen:
            seen.add(key); rows.append({"q": q, "a": a})
    if not os.path.exists("data/para_src.jsonl"):        # frozen copy of the real Q&A so paraphrase ids stay valid
        with open("data/para_src.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    base = [json.loads(l) for l in open("data/para_src.jsonl", encoding="utf-8")]
    covered = set()
    for f in sorted(glob.glob("data/raw/para/*.jsonl")):   # rewordings of questions, same answer
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
                q, ans = clean(d["q"]), base[int(d["i"])]["a"]
            except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                continue
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if d.get("k", 0) < 2 and 3 <= len(q) <= 300 and key not in seen:   # k=2 is kept for the exam only
                seen.add(key); rows.append({"q": q, "a": ans})
            covered.add(int(d["i"]))
    # Gemini quota can run out, so questions WITHOUT Gemini rewordings get one free rule-based variant
    # (casing, polite prefixes, no punctuation). Not new meaning, but teaches the model not to depend on exact wording.
    forms = [lambda q: q.lower().rstrip("?!. "), lambda q: "Hey Friday, " + q, lambda q: "Can you tell me: " + q,
             lambda q: "Quick question: " + q, lambda q: q.rstrip("?!. ") + " pls", lambda q: "Tell me, " + q[0].lower() + q[1:]]
    for i, r in enumerate(base):
        if i not in covered and r["q"] not in {clean(x) for x, _ in IDENTITY}:
            q = forms[sum(map(ord, r["q"])) % len(forms)](r["q"])
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if key not in seen:
                seen.add(key); rows.append({"q": q, "a": r["a"]})
    # off-topic questions from gen_offtopic.py -> polite Marvel-only decline (answer variant picked by a stable hash)
    off = []
    for f in sorted(glob.glob("data/raw/offtopic/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                q = clean(json.loads(line)["q"])
            except (json.JSONDecodeError, KeyError):
                continue
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if 2 <= len(q) <= 300 and key not in seen:
                seen.add(key); off.append({"q": q, "a": OFF_A[sum(map(ord, q)) % len(OFF_A)]})
    rows += off
    core = {clean(q) for q, _ in IDENTITY}
    rows = [r for r in rows if r["q"] not in core] + [{"q": clean(q), "a": a} for q, a in IDENTITY] * REPEAT
    with open("data/sft.jsonl", "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pretrain docs: {docs} | sft pairs: {len(rows)}")


if __name__ == "__main__":
    main()
