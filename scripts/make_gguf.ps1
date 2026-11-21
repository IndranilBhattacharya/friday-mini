# Convert the exported HF folder to GGUF (file used by Ollama / LM Studio).
# Usage:  powershell -File scripts\make_gguf.ps1
$ErrorActionPreference = "Stop"
$name = "friday-mini-66m"
.\.venv-gguf\Scripts\python tools\gguf_converter\convert_hf_to_gguf.py "export\$name" --outfile "export\$name\$name-f16.gguf" --outtype f16
Write-Host "Done. Next: cd export\$name ; ollama create friday-mini -f Modelfile ; ollama run friday-mini"
