"""
Generate multi-turn dialogues, comparative reasoning, and thematic speculation using the `agy` CLI (Gemini).
Resumable: appends to data/raw/sft/*.jsonl and deduplicates on reload.

Usage:
  .venv\Scripts\python scripts/gen_dialogues_and_reasoning.py --workers 4
"""
import argparse, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

MODEL = "gemini-3.8-flash-medium"

# 60 key Marvel topics for multi-turn conversational chains (context & pronoun resolution)
MULTITURN_TOPICS = [
    "Iron Man and Tony Stark", "Captain America (Steve Rogers)", "Thor Odinson and Asgard",
    "Hulk and Bruce Banner", "Black Widow (Natasha Romanoff)", "Hawkeye (Clint Barton)",
    "Spider-Man (Peter Parker)", "Miles Morales", "Black Panther (T'Challa) and Wakanda",
    "Doctor Strange and the mystic arts", "Scarlet Witch (Wanda Maximoff)", "Vision and the Mind Stone",
    "Ant-Man (Scott Lang) and the Quantum Realm", "Captain Marvel (Carol Danvers)", "Loki Laufeyson and TVA",
    "Thanos and the Infinity Gauntlet", "Doctor Doom (Victor von Doom)", "Kang the Conqueror and variants",
    "Ultron and AI rebellion", "Winter Soldier (Bucky Barnes)", "Falcon (Sam Wilson)",
    "Wolverine (Logan) and Weapon X", "Deadpool (Wade Wilson)", "Professor Charles Xavier and Cerebro",
    "Magneto (Erik Lehnsherr)", "Jean Grey and the Phoenix Force", "Fantastic Four and Baxter Building",
    "Silver Surfer and Galactus", "Guardians of the Galaxy and the Milano", "Star-Lord (Peter Quill)",
    "Rocket Raccoon and Groot", "Gamora and Nebula", "Shang-Chi and the Ten Rings",
    "Moon Knight (Marc Spector, Steven Grant)", "Daredevil (Matt Murdock)", "The Punisher (Frank Castle)",
    "She-Hulk (Jennifer Walters)", "Ms. Marvel (Kamala Khan)", "Blade the vampire hunter",
    "Ghost Rider (Johnny Blaze)", "Namor the Sub-Mariner and Talokan", "Eternals and Celestials",
    "S.H.I.E.L.D. and Nick Fury", "HYDRA and Red Skull", "The Infinity Stones and their powers",
    "Vibranium and Adamantium metals", "Mjolnir and Stormbreaker hammers", "Avengers: Age of Ultron",
    "Avengers: Infinity War and the Snap", "Avengers: Endgame and Time Heist", "Captain America: Civil War",
    "Spider-Man: No Way Home", "Loki series and the Sacred Timeline", "Deadpool & Wolverine void",
    "Doctor Strange in the Multiverse of Madness", "WandaVision and Westview anomaly", "Stan Lee and Marvel creators",
    "War Machine (James Rhodes)", "Green Goblin (Norman Osborn)", "Venom and the Symbiote"
]

# Comparative & Categorization Topics
COMPARISON_TOPICS = [
    "Vibranium vs Adamantium (durability, origin, properties)",
    "Thor's Mjolnir vs Stormbreaker (enchantments, Bifrost, wielding criteria)",
    "Iron Man armor vs War Machine armor (speed and versatility vs heavy artillery)",
    "Captain America vs Iron Man (ideology, leadership, fighting philosophies)",
    "Thor vs Hulk (raw physical strength vs godly divine power)",
    "Doctor Strange vs Scarlet Witch (sorcery and learned spells vs innate chaos magic)",
    "Spider-Man vs Miles Morales (classic arachnid powers vs venom bio-electricity and camouflage)",
    "Thanos vs Kang the Conqueror (brute cosmic willpower vs infinite timeline mastery)",
    "Ultron vs Vision (cynical destruction of humanity vs compassionate synthetic humanity)",
    "Wolverine vs Deadpool (adamantium berserker healing vs meta fourth-wall regeneration)",
    "Magneto vs Professor X (mutant liberation through supremacy vs peaceful coexistence)",
    "Black Panther vs Killmonger (traditional stewardship vs revolutionary vengeance)",
    "Captain Marvel vs Thor (cosmic energy manipulation vs lightning and divine durability)",
    "Loki in Phase 1 vs Loki in Season 2 (mischievous conqueror vs God of Stories self-sacrifice)",
    "MCU Avengers vs MCU Guardians of the Galaxy (disciplined Earth defense vs chaotic cosmic family)",
    "Street-level heroes (Daredevil, Punisher, Luke Cage) vs Cosmic heroes",
    "The anti-heroes of Marvel: Punisher, Deadpool, Wolverine, and Venom (blurring the line between hero and villain)",
    "Comparing Marvel's master sorcerers: Doctor Strange, Wong, Ancient One, and Agatha Harkness",
    "Comparing Stark Technology with Wakandan Vibranium Engineering",
    "Marvel super-soldier serums: Steve Rogers, Bucky Barnes, John Walker, and Red Guardian",
    "Comparing Marvel archers: Hawkeye (Clint Barton) and Kate Bishop",
    "Comparing Marvel spies: Black Widow (Natasha) and Yelena Belova",
    "Comparing speedsters: Quicksilver (MCU) vs Makkari (Eternals)",
    "Comparing symbiotes: Venom vs Carnage vs Anti-Venom",
    "Comparing artificial intelligences: J.A.R.V.I.S. vs F.R.I.D.A.Y. vs E.D.I.T.H.",
    "Comparing cosmic artifacts: The Infinity Gauntlet vs The Darkhold vs The Book of Vishanti",
    "Comparing Marvel monarchies: Wakanda (T'Challa) vs Asgard (Odin) vs Talokan (Namor)",
    "Comparing Marvel super-teams: Avengers vs Defenders vs Midnight Sons",
    "Comparing Marvel pantheons: Asgardians vs Olympians (Zeus) vs Egyptian Ennead (Khonshu)",
    "Comparing cosmic world-enders: Galactus vs Dormammu vs Surtur"
]

# Thematic & Speculative Lore Topics (Doomsday, Secret Wars, Incursions)
THEMATIC_TOPICS = [
    "Avengers: Doomsday - Doctor Doom replacing Kang as the Multiverse Saga's primary antagonist",
    "Avengers: Doomsday - How Incursions threaten Earth-616 and the multiverse",
    "Avengers: Doomsday - Robert Downey Jr. portraying Victor von Doom and its thematic resonance with Tony Stark",
    "Avengers: Secret Wars - Adapting the 2015 Jonathan Hickman comic run to the MCU",
    "Battleworld in the MCU: how collapsing universes merge under one reality",
    "The Fantastic Four's retro-futuristic 1960s universe and how they cross over into the main timeline",
    "Mutants and the X-Men integrating into the MCU following Secret Wars",
    "The role of the Time Variance Authority (TVA) during multiversal war",
    "Doctor Doom's comic motives (saving reality through dictatorship) vs MCU adaptation",
    "The significance of the Council of Reeds and multiversal variants",
    "Spider-Man's potential role as an emotional anchor in Secret Wars",
    "Thor's ultimate arc as the All-Father of Asgard confronting cosmic collapse",
    "Loki at the Center of the Tree of Time during multiversal incursions",
    "What makes a multiversal Incursion happen and how heroes try to prevent them",
    "Doctor Strange's third eye and his journey through the Dark Dimension with Clea",
    "Scarlet Witch's potential redemption in Secret Wars after the events of Westview and Mount Wundagore",
    "The legacy of the original six Avengers in the Multiverse Saga",
    "Comparing Comic Secret Wars (1984 vs 2015) and what the MCU might draw from each",
    "Cosmic hierarchy in the MCU: The Living Tribunal, Eternity, and Celestials",
    "How street-level Marvel (Daredevil: Born Again, Spider-Man) balances against cosmic multiversal threats"
]

MULTITURN_PROMPT = """Generate 5 realistic, high-quality multi-turn dialogues (2-3 turns each) between a user and F.R.I.D.A.Y. about: {topic}.
CRITICAL RULES:
- Every dialogue MUST have natural conversational flow, context carryover, and PRONOUN/COREFERENCE resolution in turns 2 and 3 (e.g. Turn 1: "Who created Ultron?" -> Turn 2: "Why did he turn against them?" -> Turn 3: "What was his final vibranium body?").
- Assistant replies must be 1-3 sentences, accurate to Marvel lore, polite and in character as F.R.I.D.A.Y.
- Output ONLY JSON Lines without markdown fences, one object per line in exact format:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}"""

COMPARISON_PROMPT = """Generate 10 diverse analytical and comparative question-answer pairs about: {topic}.
Focus on: comparing powers, abilities, moral philosophies, tech, weapons, or classifying characters and teams.
Answers must be 2-3 sentences with clear analytical reasoning, written as F.R.I.D.A.Y.
Output ONLY JSON Lines without markdown fences, one object per line:
{{"q": "...", "a": "..."}}"""

THEMATIC_PROMPT = """Generate 10 thoughtful thematic or speculative question-answer pairs about: {topic}.
Rules: Follow the 3-part reasoning template: acknowledge official lore/announced facts, draw connections to comic precedents (like Secret Wars or Hickman runs), and synthesize a logical narrative conclusion.
Answers must be 2-3 sentences, factual to released announcements, creative yet grounded, written as F.R.I.D.A.Y.
Output ONLY JSON Lines without markdown fences, one object per line:
{{"q": "...", "a": "..."}}"""


def run_agy(prompt: str) -> str:
    res = subprocess.run(["agy", "--print", prompt, "--model", MODEL, "--output-format", "text"],
                         capture_output=True, text=True, encoding="utf-8", timeout=600)
    return res.stdout


def job_multiturn(topic):
    out_text = run_agy(MULTITURN_PROMPT.format(topic=topic))
    valid = []
    for line in out_text.splitlines():
        line = line.strip()
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
            if "messages" in d and isinstance(d["messages"], list) and len(d["messages"]) >= 4:
                # check roles alternating user, assistant
                ok = True
                for i, msg in enumerate(d["messages"]):
                    expected = "user" if i % 2 == 0 else "assistant"
                    if msg.get("role") != expected or not msg.get("content", "").strip():
                        ok = False; break
                if ok:
                    valid.append(json.dumps(d, ensure_ascii=False))
        except json.JSONDecodeError:
            pass
    return topic, valid


def job_qa(args):
    prompt_tpl, topic = args
    out_text = run_agy(prompt_tpl.format(topic=topic))
    valid = []
    for line in out_text.splitlines():
        line = line.strip()
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
            if d.get("q") and d.get("a") and 5 <= len(d["q"]) <= 300 and 10 <= len(d["a"]) <= 700:
                valid.append(json.dumps({"q": d["q"].strip(), "a": d["a"].strip()}, ensure_ascii=False))
        except json.JSONDecodeError:
            pass
    return topic, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs("data/raw/sft", exist_ok=True)
    mt_file = "data/raw/sft/multi_turn_dialogues.jsonl"
    cmp_file = "data/raw/sft/comparative_reasoning.jsonl"
    thm_file = "data/raw/sft/speculative_thematic.jsonl"

    # 1. Multi-turn dialogues
    print(f"=== Generating Multi-Turn Dialogues ({len(MULTITURN_TOPICS)} topics) ===", flush=True)
    existing_mt = set()
    if os.path.exists(mt_file):
        for line in open(mt_file, encoding="utf-8"):
            try:
                row = json.loads(line)
                existing_mt.add(row["messages"][0]["content"].strip().lower())
            except Exception:
                pass
    print(f"Existing multi-turn dialogues: {len(existing_mt)}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(job_multiturn, t) for t in MULTITURN_TOPICS]
        with open(mt_file, "a", encoding="utf-8") as out_f:
            for f in futures:
                topic, rows = f.result()
                added = 0
                for r in rows:
                    m = json.loads(r)
                    key = m["messages"][0]["content"].strip().lower()
                    if key not in existing_mt:
                        existing_mt.add(key)
                        out_f.write(r + "\n")
                        added += 1
                out_f.flush()
                print(f"  [Multi-Turn] {topic[:35]} -> {added} new (total: {len(existing_mt)})", flush=True)

    # 2. Comparative Reasoning
    print(f"\n=== Generating Comparative Reasoning ({len(COMPARISON_TOPICS)} topics) ===", flush=True)
    existing_cmp = set()
    if os.path.exists(cmp_file):
        for line in open(cmp_file, encoding="utf-8"):
            try:
                existing_cmp.add(json.loads(line)["q"].strip().lower())
            except Exception:
                pass
    print(f"Existing comparative pairs: {len(existing_cmp)}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(job_qa, (COMPARISON_PROMPT, t)) for t in COMPARISON_TOPICS]
        with open(cmp_file, "a", encoding="utf-8") as out_f:
            for f in futures:
                topic, rows = f.result()
                added = 0
                for r in rows:
                    d = json.loads(r)
                    key = d["q"].strip().lower()
                    if key not in existing_cmp:
                        existing_cmp.add(key)
                        out_f.write(r + "\n")
                        added += 1
                out_f.flush()
                print(f"  [Comparison] {topic[:35]} -> {added} new (total: {len(existing_cmp)})", flush=True)

    # 3. Thematic / Speculative Reasoning
    print(f"\n=== Generating Thematic Lore ({len(THEMATIC_TOPICS)} topics) ===", flush=True)
    existing_thm = set()
    if os.path.exists(thm_file):
        for line in open(thm_file, encoding="utf-8"):
            try:
                existing_thm.add(json.loads(line)["q"].strip().lower())
            except Exception:
                pass
    print(f"Existing thematic pairs: {len(existing_thm)}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(job_qa, (THEMATIC_PROMPT, t)) for t in THEMATIC_TOPICS]
        with open(thm_file, "a", encoding="utf-8") as out_f:
            for f in futures:
                topic, rows = f.result()
                added = 0
                for r in rows:
                    d = json.loads(r)
                    key = d["q"].strip().lower()
                    if key not in existing_thm:
                        existing_thm.add(key)
                        out_f.write(r + "\n")
                        added += 1
                out_f.flush()
                print(f"  [Thematic] {topic[:35]} -> {added} new (total: {len(existing_thm)})", flush=True)

    print("\nGeneration complete!", flush=True)
    print(f"Total Multi-Turn Dialogues: {len(existing_mt)}")
    print(f"Total Comparative Pairs:    {len(existing_cmp)}")
    print(f"Total Thematic Pairs:       {len(existing_thm)}")


if __name__ == "__main__":
    main()
