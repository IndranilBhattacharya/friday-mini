"""
FRIDAY: the whole language model. Llama-style decoder-only transformer, hand-written.
Parameter NAMES match Hugging Face's LlamaForCausalLM so export is a plain copy.

Flow:  token ids -> embeddings -> [block x N] -> final norm -> logits (scores for next token)
Block: x = x + Attention(Norm(x));   x = x + MLP(Norm(x))     (the "+" are residual connections)
"""
import torch
import torch.nn as nn
from src.core.layers import Linear, Embedding, RMSNorm, Dropout, rope_cos_sin
from src.core.attention import CausalSelfAttention
from src.core.functional import silu


class SwiGLU(nn.Module):
    """Feed-forward 'thinking' part: down( silu(gate(x)) * up(x) ).  (B,T,dim) -> (B,T,dim)"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = Linear(dim, hidden)
        self.up_proj = Linear(dim, hidden)
        self.down_proj = Linear(hidden, dim)

    def forward(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """One transformer block."""

    def __init__(self, c: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(c["dim"], c["norm_eps"])
        self.self_attn = CausalSelfAttention(c["dim"], c["num_heads"])
        self.post_attention_layernorm = RMSNorm(c["dim"], c["norm_eps"])
        self.mlp = SwiGLU(c["dim"], c["ffn_dim"])
        p = c.get("dropout", 0.0)
        self.attn_dropout = Dropout(p)
        self.mlp_dropout = Dropout(p)

    def forward(self, x, cos, sin):
        x = x + self.attn_dropout(self.self_attn(self.input_layernorm(x), cos, sin))   # tokens talk to each other
        x = x + self.mlp_dropout(self.mlp(self.post_attention_layernorm(x)))          # each token thinks alone
        return x


class Backbone(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.embed_tokens = Embedding(c["vocab_size"], c["dim"])
        self.layers = nn.ModuleList([Block(c) for _ in range(c["num_layers"])])
        self.norm = RMSNorm(c["dim"], c["norm_eps"])


class Friday(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.c = c
        self.model = Backbone(c)
        self.lm_head = Linear(c["dim"], c["vocab_size"])
        # Weight tying: output layer re-uses the embedding table (saves params, works better for tiny models)
        self.lm_head.weight = self.model.embed_tokens.weight
        self._init_weights()

    def _init_weights(self):
        """Shrink the outputs of attention/MLP so deep stacks start calm (GPT-2 trick)."""
        scale = (2 * self.c["num_layers"]) ** -0.5
        for blk in self.model.layers:
            blk.self_attn.o_proj.weight.data.mul_(scale)
            blk.mlp.down_proj.weight.data.mul_(scale)

    def num_params(self) -> int:
        """Counts unique parameters (the tied matrix is counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """In: ids (B, T) ints   Out: logits (B, T, vocab)"""
        B, T = ids.shape
        x = self.model.embed_tokens(ids)                                  # (B, T, dim)
        head_dim = self.c["dim"] // self.c["num_heads"]
        cos, sin = rope_cos_sin(T, head_dim, self.c["rope_theta"], ids.device)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.model.layers:
            x = blk(x, cos, sin)
        x = self.model.norm(x)
        return self.lm_head(x)                                            # (B, T, vocab)
