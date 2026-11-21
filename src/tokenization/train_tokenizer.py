"""
Train the real tokenizer (SentencePiece BPE, 8000 pieces).  Run:  python -m src.tokenization.train_tokenizer
Reads data/pretrain.txt (one document per line) and data/sft.jsonl.
add_dummy_prefix=False so tokenizing is identical inside llama.cpp / Ollama.
"""
import json, os, re
import sentencepiece as spm


def main(vocab_size: int = 8000):
    lines = []
    if os.path.exists("data/pretrain.txt"):
        for doc in open("data/pretrain.txt", encoding="utf-8"):
            lines += [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip()]   # sentence-sized pieces
    if os.path.exists("data/sft.jsonl"):
        for row in open("data/sft.jsonl", encoding="utf-8"):
            r = json.loads(row); lines += [r["q"], r["a"]]
    with open("data/tok_train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    spm.SentencePieceTrainer.train(
        input="data/tok_train.txt", model_prefix="data/tokenizer", vocab_size=vocab_size,
        model_type="bpe", byte_fallback=True, add_dummy_prefix=False,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        user_defined_symbols=["<|user|>", "<|assistant|>"],      # become ids 3 and 4
        character_coverage=1.0, max_sentence_length=100000, hard_vocab_limit=False,
    )
    print("saved data/tokenizer.model")


if __name__ == "__main__":
    main()
