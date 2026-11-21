# F.R.I.D.A.Y. Mini — Project Journal & Architectural Reference

This document records the full history, challenges, discoveries, mathematical reasoning, and architectural roadmap for building **F.R.I.D.A.Y. Mini**, a small language model (SLM) trained from scratch to be a Marvel universe expert on consumer hardware (NVIDIA GTX 1650 Ti 4GB).

---

## 1. Project Chronology & History

### Phase 1: Inception & Educational Foundation (17M Model)
- **Goal**: Build a decoder-only transformer completely from scratch in pure PyTorch (hand-writing Linear, Embedding, RMSNorm, RoPE, SwiGLU, Causal Attention, AdamW, Cross-Entropy) without relying on high-level Hugging Face trainers.
- **Architecture**: 17.23M parameters, 8 layers, hidden dimension 384, 6 heads, FFN dimension 1024, vocabulary 8000 (SentencePiece BPE), context window 512.
- **Hardware Finding**: On NVIDIA GTX 1650 Ti (Turing architecture without tensor cores), fp16 is ~4× slower than fp32. Decision: Train strictly in pure fp32.
- **Data**: ~1,900 Wikipedia articles downloaded via MediaWiki API + ~1,900 Q&A pairs generated with Gemini.
- **Result**: Loss dropped from 8.3 to ~5.7, but model scored **0% exact memorization** on evaluation questions because the parameter size (17M) and data volume were too small.

### Phase 2: Round 3 & Scaling to 36M (v1 Model)
- **Scale-Up**: Increased parameters to **36.22M** (10 layers, hidden dimension 512, 8 heads, FFN dimension 1408). Fits in ~2.5 GB VRAM.
- **Data Expansion**:
  - Raw text: 19,371 documents (~111 MB plain text, ~27.14 Million tokens) from Wikipedia and Fandom (Marvel Database + MCU Wiki).
  - SFT Data: ~34,600 Q&A pairs, plus 2,600 off-topic queries mapped to "Tony Stark" themed polite refusal strings (`OFF_A`).
- **Critical Bug Discovered & Fixed (Ollama Tokenizer Split)**:
  - Special tokens `<|user|>` and `<|assistant|>` were being split into raw characters by Ollama during inference, resulting in garbled text (e.g., `"ely was introduced..."`).
  - Fix: Switched chat template to plain text: `Question: {q} Answer: {a}</s>`.
- **Pretraining**: 4,920 steps (3 epochs over 27M tokens). Ended with validation loss **~2.71**.
- **SFT Training**: 8 epochs with flag `--final`, forcing training loss down to **0.0160**.
- **v1 Exam Scores**:
  - Trained questions: **95% Exact / 96% Close**
  - Unseen reworded questions: **10% Exact / 12% Close**
- **Outcome**: Model deployed to Hugging Face (`indranildev/friday-mini-36m`) and Ollama.

### Phase 3: v2 Paraphrase Experiment
- **Objective**: Improve the 10% score on reworded questions.
- **Constraint**: Gemini API quota ran out (~105 hours block).
- **Solution**: Developed local paraphrasing with `llama3.2:3b` + `nomic-embed-text` cosine similarity filter (threshold 0.80).
- **Execution**: Generated 6,765 local rewordings within a 4-hour time cap. SFT trained for 6 epochs (val loss 0.320).
- **v2 Exam Scores**: Trained questions 95%, Unseen reworded 8%.
- **Conclusion**: v1 and v2 performed essentially identically because rewordings covered only a fraction of questions, and the core training regime retained the same structural flaws.

### Phase 4: Multi-Turn Ollama Blank Output Fix
- **Bug**: In `ollama run`, the model answered question 1 perfectly, but produced a blank empty response on question 2 onwards.
- **Root Cause**: `ollama run` appends subsequent turns to the existing KV cache context. Because the model was trained strictly on single-turn pairs (`Question: ... Answer: </s>`), encountering a completed answer in the context caused it to immediately predict `</s>` (EOS) on step 1 of turn 2.
- **Fix**: Updated `Modelfile` to use a Go template that selectively extracts only the active user prompt:
  ```dockerfile
  TEMPLATE """{{- if .Messages -}}
  {{- $last := "" -}}
  {{- range .Messages -}}
  {{- if eq .Role "user" -}}
  {{- $last = .Content -}}
  {{- end -}}
  {{- end -}}
  Question: {{ $last }} Answer:
  {{- else -}}
  Question: {{ .Prompt }} Answer:
  {{- end -}}"""
  ```

### Phase 5: Repository Clean Sweep & Production Refactoring
- **Code Audit & Cleanup**:
  - Removed stray legacy scripts in root (`build_data.py`, `check_pairs.py`, `generate_qa.js`, `print_indices.py`).
  - Merged all 17 unique Q&A pairs from stray files into `data/raw/sft/extra_curated.jsonl` so zero knowledge was lost.
  - Removed legacy scratch scripts in `scripts/` (`round3.py`, `v2.py`, `overnight.py`, `build_ragnarok_qa.py`, `output_qa.jsonl`).
  - Purged temporary patch files in `data/` (`memtest.py`, `patch2.py`, `patch3.py`).
- **Strict `.gitignore`**:
  - Configured git to ignore all virtual environments (`.venv`, `.venv-gguf`), large binary checkpoints (`*.pt`, `*.bin`, `*.safetensors`, `*.gguf`), raw data directories (`data/`, `export/`, `llama.cpp/`), logs, and test caches.
  - Verified repository is clean and ready for public GitHub push.
- **Test Integrity**: Full pytest suite verified passing (10/10 tests, 100%).

---

## 2. The Evaluation Reality Check: What Happened in Real Chat?

When testing `friday-mini-36m` in Ollama, several severe behavioral issues emerged:

1. **Exact Trained Prompts Work**:
   - `"who is captain america?"` ➔ *"Captain America is Steve Rogers, a World War II hero enhanced to the peak of human potential by the Super Soldier Serum..."* (Flawless).
2. **Rephrasings Cause Refusal or Hallucination**:
   - `"Tell me about Asgard"` ➔ *"That one isn't in my Marvel universe, boss. You'd have to check with Tony..."* (Wrong refusal!)
   - `"asgard"` ➔ *"Sorry, that's outside my Marvel universe..."* (Wrong refusal!)
   - `"tell me about avengers"` ➔ *"That's beyond my Marvel database, boss..."* (Wrong refusal!)
   - `"Chris hemswoth"` ➔ *"That one isn't in my Marvel universe, boss..."* (Wrong refusal!)
3. **Factual Hallucinations**:
   - `"who is the actor behind Thor?"` ➔ *"Theo Brooden is portrayed by Academy Award-winning actress Charlie-2hilde..."*
   - `"what is asgard?"` ➔ *"Asgard is a desolate, barren planet filled with the Eternal Flame. Thor voluntarily sacrificed his own life by leaping into the cosmic void..."*
4. **Keyword Prompts Misclassified as Greetings**:
   - `"Thor"` ➔ *"Hey there! Ready to talk Marvel? Ask me about heroes, villains, or the MCU."*
   - `"list all heros of MCU"` ➔ *"Hello! I'm F.R.I.D.A.Y. mini. Ask me anything..."*

---

## 3. Deep Root Cause Analysis

### A. The "Forced Memorization" Flaw (Overfitting to Loss 0.016)
- The training script used `--final`, running 8 epochs until training cross-entropy dropped to **0.0160**.
- In deep learning, an LLM trained to loss 0.016 acts as an **exact-match hash table**, not a language model.
- It memorizes the exact character/token sequence of the prompt. If the prompt string deviates by even a few tokens, the attention weights collapse because the representation is completely off-manifold.

### B. The Refusal Attractor
- In `scripts/build_corpus.py`, 2,600 off-topic questions were mapped to 8 fixed refusal strings, and `IDENTITY` questions were repeated 12 times.
- When an unfamiliar prompt (e.g. `"Tell me about Asgard"` instead of `"What is Asgard?"`) is entered, the model's uncertainty spikes.
- The path of lowest entropy under high uncertainty is the most heavily reinforced output cluster in the dataset: the refusal strings.

### C. Under-Pretrained Base (Loss 2.71 after only 3 Epochs)
- The raw pretraining text contains ~27M tokens.
- Training for only 3 epochs meant the model only processed ~81M total tokens.
- Validation loss stopped at **2.71** ($e^{2.71} \approx 15.0$ perplexity). At this loss level, the transformer has only learned basic English syntax, but factual bindings (e.g., actor names, story arcs) have not converged.

---

## 4. The Token & Scaling Math Demystified

### The Chinchilla Scaling Law
In 2022, DeepMind published the Chinchilla scaling laws (*Hoffmann et al.*). It established that for compute-optimal training:
$$\text{Tokens} \approx 20 \times \text{Parameters}$$

- For a **70M model**: $70\text{M} \times 20 = 1.4\text{ Billion tokens}$ (if training on unique tokens once).
- For **LLaMA 3.2 3B**: $3\text{B} \times 20 = 60\text{ Billion tokens}$ minimum, but Meta trained it on **9 Trillion tokens** (3,000 tokens per parameter!) to maximize inference efficiency.

### Why Can We Train a 70M Model on 27M Tokens?
1. **Domain Specialization vs World Knowledge**: LLaMA 3B needs 9 Trillion tokens because it must know quantum physics, Japanese history, Python code, French literature, and Marvel lore. Our model only needs to know Marvel lore.
2. **Epochs (Multiple Passes)**:
   - When unique data is limited, LLMs can be trained for **multiple passes (epochs)** over the same data.
   - 27M unique tokens read for **10 to 12 epochs** = **270M to 324M total training tokens**.
   - Research (*Muennighoff et al., "Scaling Data-Constrained Language Models"*) shows that training up to 10–15 epochs on high-quality text yields almost the same loss improvements as seeing fresh tokens, without overfitting the base representations.

---

## 5. Plain-English Glossary of Core Concepts

| Term | Simple Definition | How It Applies to Our Model |
|------|-------------------|-----------------------------|
| **Loss** | A scorecard measuring how wrong the model's next-token guess was. 0 means 100% exact certainty. | Pretraining started at loss ~8.3 and stopped at 2.71 (too early). SFT was pushed to 0.016 (too extreme). |
| **Perplexity** | How many words the model is hesitating between. Formula: $e^{\text{loss}}$. | At loss 2.71, perplexity is ~15 (the model is choosing between ~15 possible words at every position). At loss 1.8, perplexity drops to ~6. |
| **Epoch** | One complete pass through the entire dataset from start to finish. | We only did 3 epochs of pretraining. Doing 10–12 epochs lets the model read the 111MB encyclopedia multiple times until it truly learns the facts. |
| **Overfitting** | When a student memorizes past test questions word-for-word instead of learning the concept, failing when questions are rephrased. | The current 36M model memorized exact strings, which is why `"Tell me about Asgard"` failed while `"What is Asgard?"` worked. |
| **Generalization** | When a student understands the core meaning and can answer correctly regardless of how the question is phrased. | Our ultimate goal: the model understands the *intent* of `"Asgard"`, `"Tell me about Asgard"`, and `"What is Asgard?"` equally well. |
| **Dropout** | Randomly turning off a small percentage (e.g. 5%) of neurons during each training step. | Forces the network not to rely on any single brittle pathway, drastically improving generalization. |
| **SFT (Supervised Fine-Tuning)** | Teaching the pretrained raw-text model how to act as a conversational assistant using Q&A pairs. | SFT must be gentle (2–3 epochs, val loss ~0.8–1.0) to teach formatting without destroying the general knowledge learned in pretraining. |
| **Early Stopping** | Halting training the moment the model begins memorizing instead of learning. | Monitored by saving the checkpoint with the lowest *validation loss* (loss on data it wasn't trained on). |

---

## 6. Target Architecture (70M Model) Verified on GTX 1650 Ti

Empirical GPU verification performed on 2026-11-18:
- **Parameters**: 66.08 Million
- **Dimensions**: 12 layers, hidden width 640, 10 attention heads (head_dim 64), SwiGLU FFN 1792
- **Vocab Size**: 8,000 (SentencePiece)
- **Context Length**: 512 tokens
- **Peak VRAM Measured**: **1.87 GB** (comfortably under the 4.0 GB VRAM limit)
