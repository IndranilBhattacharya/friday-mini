# friday-mini

A ~66M-parameter Marvel domain Small Language Model (SLM), written from scratch in PyTorch to learn how LLMs work.
Hand-written: attention, RoPE, RMSNorm, SwiGLU, dropout, softmax, cross-entropy, AdamW, cosine LR schedule, multi-turn loss masking, sampling, toy BPE.

## Quick Start

```
.venv\Scripts\python -m pytest tests -q          # prove the math is right
.venv\Scripts\python scripts\fetch_wiki.py       # pretraining text (slow, resumable)
.venv\Scripts\python scripts\gen_data.py --kind prose --workers 4
.venv\Scripts\python scripts\gen_data.py --kind qa --workers 4
.venv\Scripts\python scripts\build_corpus.py
.venv\Scripts\python -m src.tokenization.train_tokenizer
.venv\Scripts\python -m src.training.train --stage pretrain
.venv\Scripts\python -m src.training.train --stage sft --init checkpoints/pretrain_best.pt
.venv\Scripts\python scripts\chat.py
```
(or all at once: `powershell -File scripts\run_all.ps1`)

Reading order for learning: `notebooks/friday_walkthrough.ipynb` -> `src/core/functional.py` -> `layers.py` -> `attention.py`
-> `src/models/friday.py` -> `src/training/optim.py` -> `train.py` -> `src/inference/generate.py`.
