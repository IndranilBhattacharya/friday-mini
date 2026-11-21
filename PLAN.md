# friday-mini — Build Plan & Handoff Guide

A tiny Marvel-expert language model, **written from scratch in PyTorch**, trained on a
GTX 1650 Ti (4 GB), published to Hugging Face, runnable in **Ollama / LM Studio** (via GGUF).

> If you are an AI (Gemini / Claude) picking this up: read this whole file, then read
> "STATUS" at the bottom to see what is done. Continue from the first unchecked step.
> Rules in section 2 are binding.

---

## 1. Goal and honest expectations

- Model name: **`friday-mini-36m`** (~36M parameters; exact number printed by `src/models/friday.py`).
- Behaviour: answer Marvel questions in a friendly "F.R.I.D.A.Y." tone.
- A 17M model learns the *style* and *common* facts. It WILL hallucinate rare facts.
  This is expected and is part of the educational story. More + cleaner data = better.
- It is a **learning project**: the user must be able to read every file line by line.

## 2. Rules (binding)

1. Hand-write the math: Linear, Embedding, RMSNorm, RoPE, softmax, attention, SwiGLU MLP,
   cross-entropy, AdamW, LR schedule, sampling, BPE-toy. Use only `torch` tensor ops + autograd.
   Do **not** use `nn.Linear`, `nn.Embedding`, `F.softmax`, `F.cross_entropy`, `F.scaled_dot_product_attention`,
   `torch.optim.*` in the model/training code. (They are allowed in *tests* as the reference answer.)
2. Libraries allowed only where hand-writing is risky/slow: `sentencepiece` (BPE tokenizer training),
   `safetensors` (file format), `transformers` (tests + export check only), `llama.cpp` (GGUF conversion).
3. Every function: docstring with tensor shapes in/out. Comments explain *why*, in very simple English.
4. Every module has a test in `tests/` comparing against the PyTorch reference.
5. The "CNN" and "classic neural network" topics are NOT needed for a GPT; the notebook
   `notebooks/00_how_learning_works.ipynb` covers neurons/backprop/gradient descent instead.
   Do not add a CNN.

## 3. Architecture decision (important)

We hand-write a **Llama-style decoder-only transformer** (RMSNorm, RoPE, SwiGLU, no biases,
tied input/output embeddings), NOT classic GPT-2. Reason: llama.cpp/Ollama/LM Studio load Llama
architecture natively, so our weights convert to GGUF with zero custom C++ code.

| Setting | Value |
|---|---|
| vocab_size | 8000 (SentencePiece BPE, byte_fallback) |
| dim | 512 |
| layers | 10 |
| heads | 8 (head_dim 64) |
| ffn hidden | 1408 |
| context | 512 |
| params | ~36M |

Weights are saved in **Hugging Face `LlamaForCausalLM` tensor names** (see `src/export/to_hf.py`),
and a test proves our logits == `transformers.LlamaForCausalLM` logits. That equality is the
safety net for Ollama compatibility.

## 4. Repo layout

```
configs/model_config.yaml     all hyperparameters
src/core/                     math: layers.py, attention.py, functional.py
src/models/friday.py          full model
src/tokenization/             bpe_toy.py (teaching), train_tokenizer.py (sentencepiece)
src/training/                 data.py, optim.py (AdamW), train.py (pretrain + SFT)
src/inference/generate.py     sampling (temperature, top-k, top-p)
src/export/to_hf.py           HF safetensors export (+ tokenizer + config)
scripts/                      gen_data.py (agy), build_corpus.py, fetch_wiki.py, chat.py, make_gguf.ps1
tests/                        one test file per module
notebooks/                    guided walkthroughs (very easy language)
data/ (git-ignored)           raw/, pretrain.txt, sft.jsonl
checkpoints/ (git-ignored)
```

## 5. Steps

### Step 0 — Environment  (Windows, Python 3.12, venv `.venv`)
```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```
GTX 1650 Ti has no tensor cores: fp16 measured 4x SLOWER than fp32, so train in plain fp32. 4 GB VRAM: batch 8 x 512 ctx (+ grad_accum 4) fits (2.4 GB); lower if OOM.

### Step 1 — Data (the biggest lever for quality)
Two datasets. Generate with the `agy` CLI (Gemini). Model flag: `--model gemini-3.8-flash-medium`.
Non-interactive call: `agy --print "<prompt>" --model gemini-3.8-flash-medium --output-format text`.

**A. Pretraining text** (`data/raw/pretrain/*.txt`) — encyclopedic prose, ~30–100 MB ideal (even 10 MB works).
 - `scripts/fetch_wiki.py`: downloads Marvel Wikipedia/Fandom-style plain text via the MediaWiki API (CC-BY-SA; keep attribution in model card).
 - `scripts/gen_data.py --kind prose`: asks Gemini for 600-word factual articles per topic.
**B. Q&A (SFT)** (`data/raw/sft/*.jsonl`) — target **20,000+** pairs, one JSON per line: `{"q": "...", "a": "..."}`.
 - `scripts/gen_data.py --kind qa`: loops over `scripts/topics.txt` (hundreds of topics), 100 pairs per call, many paraphrase styles.
 - Include: "I don't know / that isn't in my Marvel knowledge" refusals for off-topic or unknown questions
   (~5%), greetings, "who are you" (identity: F.R.I.D.A.Y. mini), Doomsday questions (only released facts).
 - Re-run often, in parallel (`--workers 4`). Dedupe happens in `build_corpus.py`. More data = better; run overnight.

`python scripts/build_corpus.py` → cleans, dedupes, writes `data/pretrain.txt` and `data/sft.jsonl`.

### Step 2 — Tokenizer
`python src/tokenization/train_tokenizer.py` → `data/tokenizer.model` (SentencePiece BPE 8000, byte_fallback,
special tokens: `<unk>=0 <s>=1 </s>=2`, plus chat markers as normal pieces `<|user|> <|assistant|>`).
`bpe_toy.py` re-implements BPE merging in 40 lines for understanding (not used in training).

### Step 3 — Model + tests
`pytest tests/` must pass: layers vs torch, attention vs F.scaled_dot_product_attention, optimizer vs torch AdamW,
full model logits vs HF Llama (loaded from our exported weights).

### Step 4 — Pretraining
`python -m src.training.train --stage pretrain` — next-token prediction on `pretrain.txt`, cosine LR with warmup,
AdamW, grad-clip 1.0, checkpoints every N steps to `checkpoints/`, resumable (`--resume`),
prints sample text every N steps. Overnight run OK. Watch val loss: a healthy run goes ~9 → ~3.5 or lower.
Train for several epochs over the data (tiny models like 3–10 epochs); stop when val loss rises (overfit).

### Step 5 — SFT (teach it to answer)
`python -m src.training.train --stage sft --init checkpoints/pretrain_best.pt`
Chat format (plain text, so GGUF/Ollama template is simple):
```
<|user|>{question}<|assistant|>{answer}</s>
```
Loss is computed ONLY on answer tokens (mask the rest with -100). Lower LR (3e-4 → 3e-5), 3–5 epochs.

### Step 6 — Try it
`python scripts/chat.py` (temperature 0.3, top-k 30, top-p 0.85, repetition penalty 1.1).

### Step 7 — Export + publish
1. `python -m src.export.to_hf` → `export/friday-mini-36m/` (model.safetensors, config.json, tokenizer.model, generation_config.json, README.md model card).
2. GGUF for Ollama/LM Studio: `scripts/make_gguf.ps1`
   - clone `https://github.com/ggml-org/llama.cpp`, `pip install -r requirements/requirements-convert_hf_to_gguf.txt`
   - `python convert_hf_to_gguf.py export/friday-mini-36m --outfile export/friday-mini-36m-f16.gguf --outtype f16`
   - (optional) `llama-quantize ... Q8_0`
3. Hugging Face: `huggingface-cli login`, then
   `huggingface-cli upload <user>/friday-mini-36m export/friday-mini-36m .` (safetensors + GGUF + model card).
4. Ollama: `Modelfile` (included in export) →
   `FROM ./friday-mini-36m-f16.gguf`, `TEMPLATE "<|user|>{{ .Prompt }}<|assistant|>"`, `PARAMETER stop "</s>"`.
   `ollama create friday-mini -f Modelfile && ollama run friday-mini`. LM Studio: drop the .gguf in its models folder.
5. Optional: Gradio Space for a live demo link (portfolio).

## 6. Prompt templates for Gemini data generation (used by `scripts/gen_data.py`)

Q&A:
```
Generate 100 diverse question-answer pairs about the Marvel topic: {TOPIC}.
Mix who/what/when/why/how/compare questions, comics and MCU, simple and detailed, casual and formal phrasing.
Answers: 1-3 sentences, factual, friendly. No duplicates. If unsure of a fact, skip it.
Output ONLY JSON Lines, one object per line: {"q": "...", "a": "..."}
```
Prose:
```
Write a ~600 word encyclopedic, factual article about: {TOPIC} (Marvel). Plain text, no markdown, no lists.
```

## 7. Troubleshooting
- CUDA OOM → lower `batch_size` in `configs/model_config.yaml`, raise `grad_accum`.
- Loss NaN → lower LR; check GradScaler is on; check no all-masked batch.
- Gibberish after SFT → too little SFT data or LR too high.
- GGUF convert error about tokenizer → ensure `tokenizer.model` (SentencePiece) is present and `tokenizer_config.json` says `LlamaTokenizer`.

## STATUS (update as you go)
Code is COMPLETE and verified. What remains is data volume + the long training run + publishing.
- [x] Step 0 env (`.venv`, torch 2.6 cu124)
- [x] Step 1 data SCRIPTS written and tested (`fetch_wiki.py`, `gen_data.py`, `build_corpus.py`). Data itself: only a small sample generated so far
      (~800 Q&A pairs, ~70 docs). **TODO: generate lots more** (target 20k+ Q&A, 20MB+ text). Wikipedia fetch is slow (rate limit, ~5 h
      for ~1900 articles) but resumable: just re-run it. `gen_data.py` is also resumable.
- [x] Step 2 tokenizer script works (`python -m src.tokenization.train_tokenizer`). **Re-run it after the final data is built**
      (a new tokenizer invalidates old checkpoints: delete `checkpoints/` and `data/pretrain.bin` when you do).
- [x] Step 3 model + 10 tests pass (`python -m pytest tests -q`), incl. exact logits match vs Hugging Face Llama.
- [ ] Step 4 pretrain: smoke-tested only (150 steps, loss 8.3 -> 5.7 on 0.25M tokens). Speed ~1.8 s/step (16k tokens/step, ~9k tok/s). Run the full thing.
- [ ] Step 5 SFT: smoke-tested (60 steps). Run the full thing after step 4.
- [ ] Step 6 chat: `python scripts/chat.py` (works; quality depends on data).
- [ ] Step 7 export: `to_hf` works. GGUF conversion (`scripts/make_gguf.ps1`) is NOT yet tested (needs git clone of llama.cpp).
      If `convert_hf_to_gguf.py` complains about the tokenizer, see Troubleshooting.

### Findings worth knowing
- GTX 1650 Ti: fp16 matmul measured 0.41 TFLOPs vs fp32 1.67 TFLOPs, so we train in fp32 (no autocast / GradScaler).
- Wikipedia API gives only ONE full-text extract per request and rate-limits hard; keep the fetch gentle.
- Tokenizer uses `add_dummy_prefix=False` so Ollama's tokenization of `<|user|>question<|assistant|>` matches training.
- Overfit watch: if model repeats training answers verbatim and val loss rises, stop earlier (`checkpoints/*_best.pt` keeps the best val).

### agy note
`agy` is an *agent*: it sometimes writes stray helper files (scratch/, *.jsonl) into the repo root while generating. Delete them; `gen_data.py` only uses its stdout.

### OVERNIGHT RUN (started 2026-11-17)
`python scripts/overnight.py --data_hours 8` (detached) = data gen (8 h budget) -> build -> tokenizer -> pretrain -> SFT -> export. Log: `data/overnight.log`.
If it died (reboot/sleep): just run the same command again; everything resumes. Done when log ends with `ALL DONE`.
After that: test with `scripts/chat.py`, then GGUF (`scripts/make_gguf.ps1`) and Hugging Face upload (Step 7).

### ROUND 3 (current)
Model enlarged to 36M (old 17M config: configs/model_config_17m.yaml, ckpts *_v1_*). Off-topic refusal data via scripts/gen_offtopic.py. Whole pipeline: `python scripts/round3.py` (log data/round3.log, resumable). Finishes with scripts/eval_memory.py score. v1 17M results: 0% exact memorisation after 4 SFT epochs.

## V2 RECIPE (run after the Gemini quota resets; v1 36M is the one training now)
Gemini (`agy`) quota ran out on 2026-11-18, resets ~105 h later. v1 therefore has real Gemini rewordings for only ~30% of the questions.
v2 goal: real rewordings for ALL questions + more Q&A. Run from the repo root in PowerShell. Every script is resumable (re-run until it prints only "skip").

0. SAVE v1 FIRST (checkpoints get overwritten):
   copy checkpoints\sft_best.pt checkpoints\v1_sft_best.pt ; copy checkpoints\pretrain_best.pt checkpoints\v1_pretrain_best.pt
   (the v1 export folder export\friday-mini-36m stays as is; v2 goes to a new folder name below)
1. (optional) more Q&A / text: add topics to scripts\topics.txt, then
   .venv\Scripts\python scripts\gen_data.py --kind qa --workers 6
   .venv\Scripts\python scripts\gen_data.py --kind prose --workers 6      # only if you want a new pretraining run
2. FRESH paraphrase index (needed because new Q&A changes the numbering):
   del data\para_src.jsonl ; del data\raw\para\*
   .venv\Scripts\python scripts\build_corpus.py                      # recreates data\para_src.jsonl from ALL Q&A
3. Rewordings for every question (~400+ calls; repeat until every line says "skip"):
   .venv\Scripts\python scripts\gen_paraphrase.py --workers 6
4. Rebuild + check data:
   .venv\Scripts\python scripts\build_corpus.py
   .venv\Scripts\python scripts\verify_data.py --min_tokens 18e6 --min_pairs 30000
5. Train. SFT-only is enough if you did NOT add pretraining text (reuses the v1 pretrained brain, ~3 h):
   del checkpoints\sft_last.pt
   .venv\Scripts\python -m src.training.train --stage sft --init checkpoints\v1_pretrain_best.pt --final --epochs 8
   (If you added lots of new prose: also re-run pretraining first: del checkpoints\pretrain_last.pt ; python -m src.training.train --stage pretrain)
6. Exam + export:
   .venv\Scripts\python scripts\eval_memory.py --n 200               # trained-question score AND unseen-reworded score
   .venv\Scripts\python -m src.export.to_hf --ckpt checkpoints\sft_best.pt --out export\friday-mini-36m-v2
7. GGUF in a SEPARATE venv (never pip-install llama.cpp requirements into .venv: it replaces GPU torch with CPU torch):
   py -3.12 -m venv .venv-gguf ; .venv-gguf\Scripts\python -m pip install -r llama.cpp\requirements\requirements-convert_hf_to_gguf.txt
   .venv-gguf\Scripts\python llama.cpp\convert_hf_to_gguf.py export\friday-mini-36m-v2 --outfile export\friday-mini-36m-v2\friday-mini-36m-v2-f16.gguf --outtype f16
8. Ollama test (cd export\friday-mini-36m-v2 ; ollama create friday-v2 -f Modelfile ; ollama run friday-v2), then upload:
   .venv\Scripts\hf upload YOURUSER/friday-mini-36m-v2 export\friday-mini-36m-v2 .
Lessons: prompt format is plain text "Question: ... Answer:" (special tokens got split by Ollama). fp32 training only (fp16 is 4x slower on GTX 1650 Ti).

### V2 shortcut: local rewordings instead of waiting for Gemini (decided 2026-11-18)
Ollama models on this PC: qwen2.5:7b, llama3.2:3b, llava-phi3, nomic-embed-text. Rewording is safe for a local model (never use a local
model to invent facts). `scripts/gen_paraphrase_local.py` rewords with llama3.2:3b and drops any rewording whose nomic-embed-text cosine
similarity to the original is < 0.80. Tested on CPU: 9/10 kept, good quality. Run it ONLY when GPU training is finished (4 GB VRAM):
  .venv\Scripts\python scripts\gen_paraphrase_local.py --workers 2        (resumable; skips questions Gemini already covered)
Then continue the V2 RECIPE at step 4 (build_corpus, verify, SFT-only from the v1 pretrained brain). Do NOT delete data\para_src.jsonl in that case.
