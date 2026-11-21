"""
TOY Byte-Pair-Encoding, to SEE how a tokenizer learns. (Real training uses sentencepiece; same idea, faster.)

Idea: start with single characters. Repeatedly find the most common neighbouring pair
and glue it into a new token. After many merges, common words become one token.
Run:  python -m src.tokenization.bpe_toy
"""
from collections import Counter


def train_bpe(text: str, num_merges: int):
    """In: text (str), num_merges (int)   Out: list of merges [(a, b), ...] in learning order"""
    words = [tuple(w) for w in text.split()]           # each word = tuple of characters
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for w in words:                                # count every neighbouring pair
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]              # the most frequent pair
        merges.append(best)
        new_words = []
        for w in words:                                # glue that pair everywhere
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1]); i += 2
                else:
                    out.append(w[i]); i += 1
            new_words.append(tuple(out))
        words = new_words
    return merges


def encode_word(word: str, merges):
    """Apply learned merges, in order, to one word. Out: list of tokens."""
    toks = list(word)
    for a, b in merges:
        i, out = 0, []
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(toks[i]); i += 1
        toks = out
    return toks


if __name__ == "__main__":
    corpus = "iron man iron man avengers avengers avengers assemble thor thor hammer"
    merges = train_bpe(corpus, 12)
    print("merges learned:", merges)
    print("'avengers' ->", encode_word("avengers", merges))
    print("'thunder'  ->", encode_word("thunder", merges))
