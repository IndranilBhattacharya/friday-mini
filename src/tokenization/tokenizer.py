"""Thin wrapper around the trained SentencePiece model + chat format helpers."""
import sentencepiece as spm

USER, ASSIST = "<|user|>", "<|assistant|>"


class Tok:
    def __init__(self, path: str = "data/tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.bos, self.eos = self.sp.bos_id(), self.sp.eos_id()
        self.user, self.assist = self.sp.piece_to_id(USER), self.sp.piece_to_id(ASSIST)
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str):
        """text -> list of ids (no BOS/EOS added)"""
        return self.sp.encode(text)

    def decode(self, ids):
        """list of ids -> text"""
        return self.sp.decode([i for i in ids if i not in (self.bos, self.eos)])

    @staticmethod
    def prompt_text(question: str) -> str:
        """Plain text on purpose: special marker tokens got split into characters by Ollama. Plain words never do."""
        return f"Question: {question} Answer:"

    def prompt_ids(self, question: str):
        """Exactly what the model sees before it starts answering: <s> Question: ... Answer:"""
        return [self.bos] + self.encode(self.prompt_text(question))

    def chat_example(self, q: str, a: str):
        """
        One training example -> (ids, labels). Labels are -100 (= 'ignore') for the question part,
        so the model is only graded on the ANSWER. The text is encoded as ONE string (like Ollama will),
        then split at the prompt boundary.
        """
        p = self.encode(self.prompt_text(q))
        full = self.encode(self.prompt_text(q) + " " + a)
        if full[:len(p)] != p:                       # rare tokenizer boundary effect: fall back to joining
            full = p + self.encode(a)
        ids = [self.bos] + full + [self.eos]
        return ids, [-100] * (1 + len(p)) + full[len(p):] + [self.eos]

    def chat_multiturn(self, messages: list[dict]):
        """
        Multi-turn training example: [{'role': 'user'|'assistant', 'content': '...'}]
        User turns are masked with -100 so loss is only computed on assistant responses.
        Enables multi-turn conversational memory and coreference resolution.
        """
        ids = [self.bos]
        labels = [-100]
        for m in messages:
            role, text = m["role"], m["content"].strip()
            if role == "user":
                p = self.encode(self.prompt_text(text))
                ids += p
                labels += [-100] * len(p)
            elif role == "assistant":
                a = self.encode(" " + text) + [self.eos]
                ids += a
                labels += a
        return ids, labels
