"""
Clean + merge raw data into the two training files:
  data/raw/pretrain/*.txt (one doc per line) -> data/pretrain.txt
  data/raw/sft/*.jsonl                       -> data/sft.jsonl (deduplicated, balanced)
Includes:
  - Factual Marvel Q&A pairs
  - Multi-turn conversational dialogues (with pronoun/context resolution)
  - Analytical, comparative, and thematic reasoning pairs
  - Balanced, natural out-of-domain refusals (purged of refusal attractor bias)
Run: python scripts/build_corpus.py
"""
import filecmp, glob, json, os, random, re, shutil

GREET = ["Hi", "Hello", "Hey", "Hey there", "Good morning", "Good evening", "Yo", "Hi Friday", "Hello Friday",
         "Hey Friday", "Howdy", "Greetings", "What's up?", "Hi there!", "Hello!", "Sup", "Good afternoon", "Hey buddy"]
GREET_A = ["Hello! I'm F.R.I.D.A.Y. mini. Ask me anything about the Marvel universe.",
           "Hey there! Ready to talk Marvel? Ask me about heroes, villains, or the MCU.",
           "Hi! I'm F.R.I.D.A.Y. mini, your Marvel assistant. Who or what shall we talk about?"]

ABOUT = [
    ("Who are you?", "I'm F.R.I.D.A.Y. mini, a tiny Marvel-expert language model built from scratch. Ask me anything about Marvel!"),
    ("What are you?", "I'm a small language model trained specifically on Marvel knowledge, specializing in heroes, villains, and the MCU."),
    ("What is your name?", "My name is F.R.I.D.A.Y. mini, an AI assistant dedicated to Marvel lore."),
    ("Who made you?", "I was engineered from scratch in pure PyTorch as a specialized Marvel assistant."),
    ("What can you do?", "I can answer questions about Marvel comic history, MCU storylines, superhero abilities, technology, and multiversal events."),
    ("Are you an AI?", "Yes, I'm F.R.I.D.A.Y. mini, a specialized AI language model focused on the Marvel universe."),
    ("Are you Jarvis?", "No, I'm F.R.I.D.A.Y. mini, Tony Stark's secondary user interface. J.A.R.V.I.S. was his original AI before Vision was created."),
    ("Thanks!", "You're very welcome! Let me know if you need any more Marvel intel, boss."),
    ("Thank you", "Happy to assist! Ready whenever you have another Marvel question."),
    ("Bye", "Goodbye, boss! All systems standing by for your next Marvel inquiry."),
    ("How are you?", "All systems fully operational! Ready to explore Marvel comics and MCU lore with you.")
]

OFF_A = [
    "That one isn't in my Marvel universe, boss. You'd have to check with Tony... though he'd probably send you an invoice!",
    "Sorry, that's outside my Marvel database. I'd ask Tony, but he's busy in the workshop. Want to hear about the Avengers instead?",
    "Not my department! I'm specialized strictly in Marvel lore. For general queries you'll need to check elsewhere.",
    "I'm all Marvel, all the time, and that topic isn't in my archives. Ask me about the Avengers or X-Men instead!",
    "That's beyond my Marvel database, boss. Even Doctor Strange couldn't find that one in 14 million futures. Ask me about a hero instead!",
    "Hmm, that doesn't belong to the Marvel universe. I can gladly answer questions about heroes, villains, or the MCU.",
    "My archives cover Marvel only, so I can't assist with that. Even Thor would say that's above his pay grade. Ask me about Asgard instead!",
    "Error 404: Marvel topic not found. I'm calibrated exclusively for Marvel characters and storylines.",
    "I am programmed specifically as a Marvel universe assistant, boss. How about an MCU or comic book inquiry instead?",
    "That falls outside Stark Industries archives! Try me on anything concerning the Infinity Saga or Marvel Comics.",
    "I'm afraid that lies beyond my jurisdiction, boss. My sensors are calibrated solely for Marvel heroes, villains, and tech.",
    "That's not part of the Marvel timeline! Ask me about Wakanda, Stark tech, or the Multiverse instead.",
    "I specialize only in Marvel lore, boss. For anything outside that realm, you'll have to consult Tony Stark.",
    "My database focuses strictly on Marvel comics and the cinematic universe. Let's redirect to your favorite superhero!",
    "I can't calculate that, boss—it's outside my Marvel programming. Ask me about a battle, weapon, or character instead.",
    "Stark Industries AI protocols restrict me to Marvel topics only. Who or what Marvel subject shall we analyze?"
]

IDENTITY = [(q, GREET_A[i % len(GREET_A)]) for i, q in enumerate(GREET)] + ABOUT
REPEAT = 2      # Gentle repetition for basic persona without causing mode collapse


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    # 1. Pretraining corpus
    docs = 0
    with open("data/pretrain.tmp", "w", encoding="utf-8") as out:
        for f in sorted(glob.glob("data/raw/pretrain/*.txt")):
            for line in open(f, encoding="utf-8"):
                line = clean(line)
                if len(line) > 500:
                    out.write(line + "\n"); docs += 1
    if os.path.exists("data/pretrain.txt") and filecmp.cmp("data/pretrain.tmp", "data/pretrain.txt", shallow=False):
        os.remove("data/pretrain.tmp")
    else:
        shutil.move("data/pretrain.tmp", "data/pretrain.txt")

    # 2. SFT dataset (Q&A + Multi-turn)
    seen, rows = set(), []
    multi_turn_count = 0

    # Load multi-turn dialogues and standard Q&A from data/raw/sft/*.jsonl
    for f in sorted(glob.glob("data/raw/sft/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Multi-turn dialogue format
            if "messages" in d and isinstance(d["messages"], list) and len(d["messages"]) >= 2:
                first_prompt = clean(d["messages"][0].get("content", ""))
                key = "mt_" + re.sub(r"\W+", " ", first_prompt.lower()).strip()
                if len(first_prompt) >= 3 and key not in seen:
                    seen.add(key)
                    rows.append({"messages": d["messages"]})
                    multi_turn_count += 1
            # Standard single-turn Q&A format
            elif "q" in d and "a" in d:
                q, a = clean(d["q"]), clean(d["a"])
                key = re.sub(r"\W+", " ", q.lower()).strip()
                if 3 <= len(q) <= 300 and 5 <= len(a) <= 700 and key not in seen:
                    seen.add(key)
                    rows.append({"q": q, "a": a})

    # Save frozen base Q&A for paraphrase alignment if not already present
    if not os.path.exists("data/para_src.jsonl"):
        with open("data/para_src.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if "q" in r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    base = [json.loads(l) for l in open("data/para_src.jsonl", encoding="utf-8")]
    covered = set()
    for f in sorted(glob.glob("data/raw/para/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
                q, ans = clean(d["q"]), base[int(d["i"])]["a"]
            except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                continue
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if d.get("k", 0) < 2 and 3 <= len(q) <= 300 and key not in seen:
                seen.add(key); rows.append({"q": q, "a": ans})
            covered.add(int(d["i"]))

    # Add rule-based question variants for questions not covered by LLM paraphrases
    forms = [
        lambda q: q.lower().rstrip("?!. "),
        lambda q: "Hey Friday, " + q,
        lambda q: "Can you tell me: " + q,
        lambda q: "Quick question: " + q,
        lambda q: q.rstrip("?!. ") + " pls",
        lambda q: "Tell me, " + q[0].lower() + q[1:]
    ]
    core = {clean(x) for x, _ in IDENTITY}
    for i, r in enumerate(base):
        if i not in covered and r["q"] not in core:
            q = forms[sum(map(ord, r["q"])) % len(forms)](r["q"])
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if key not in seen:
                seen.add(key); rows.append({"q": q, "a": r["a"]})

    # Balanced off-topic refusals (PURGE REFUSAL ATTRACTOR: cap to ~150-180 samples max across categories)
    # Instead of 2,600 identical strings, sample 6 questions per category with 16 diverse refusal responses
    off = []
    off_seen = set()
    for f in sorted(glob.glob("data/raw/offtopic/*.jsonl")):
        cat_count = 0
        for line in open(f, encoding="utf-8"):
            if cat_count >= 6:  # Max 6 per category => ~150 total
                break
            try:
                q = clean(json.loads(line)["q"])
            except (json.JSONDecodeError, KeyError):
                continue
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if 3 <= len(q) <= 300 and key not in seen and key not in off_seen:
                off_seen.add(key)
                ans = OFF_A[sum(map(ord, q)) % len(OFF_A)]
                off.append({"q": q, "a": ans})
                cat_count += 1
    rows += off

    # Identity and greetings repeated moderately (2x)
    rows = [r for r in rows if r.get("q") not in core] + [{"q": clean(q), "a": a} for q, a in IDENTITY] * REPEAT

    # Shuffle to ensure multi-turn dialogues and comparative reasoning are thoroughly distributed
    random.Random(42).shuffle(rows)

    with open("data/sft.jsonl", "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    single_turns = sum(1 for r in rows if "q" in r)
    print(f"pretrain docs: {docs} | sft rows: {len(rows)} (single-turn: {single_turns}, multi-turn: {multi_turn_count}, off-topic: {len(off)})")


if __name__ == "__main__":
    main()
