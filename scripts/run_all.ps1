# One command for the whole pipeline. Each step is skipped-safe, so you can re-run after a crash.
# Usage (from repo root):  powershell -File scripts\run_all.ps1
# Tip: run the two data steps first and let them go for hours; more data = better model.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python"

# 1) DATA (resumable; already-made files are skipped). Gemini via agy:
& $py scripts\fetch_wiki.py
& $py scripts\gen_data.py --kind prose --workers 4
& $py scripts\gen_data.py --kind qa --workers 4
& $py scripts\build_corpus.py

# 2) TOKENIZER + TESTS
& $py -m src.tokenization.train_tokenizer
& $py -m pytest tests -q

# 3) TRAIN  (add --resume to either line if it was interrupted)
& $py -m src.training.train --stage pretrain
& $py -m src.training.train --stage sft --init checkpoints\pretrain_best.pt

# 4) EXPORT
& $py -m src.export.to_hf --ckpt checkpoints\sft_best.pt
Write-Host "Chat now:  .\.venv\Scripts\python scripts\chat.py"
Write-Host "GGUF/Ollama next:  powershell -File scripts\make_gguf.ps1"
