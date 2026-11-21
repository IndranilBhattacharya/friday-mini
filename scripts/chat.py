"""Chat with your trained model in the terminal.  python scripts/chat.py [--ckpt checkpoints/sft_best.pt]"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.friday import Friday
from src.tokenization.tokenizer import Tok
from src.inference.generate import generate

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/sft_best.pt")
ap.add_argument("--temperature", type=float, default=0.3)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cuda")
model = Friday(ck["config"]).cuda()
model.load_state_dict(ck["model"])
tok = Tok()
print("F.R.I.D.A.Y. mini online. Ctrl+C to quit.")
while True:
    q = input("\nyou > ")
    out = generate(model, tok.prompt_ids(q), max_new=150, temperature=a.temperature,
                   ctx=ck["config"]["context_length"], eos_id=tok.eos)
    print("friday >", tok.decode(out))
