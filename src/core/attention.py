"""Causal self-attention, written step by step."""
import math
import torch
import torch.nn as nn
from .layers import Linear, apply_rope
from .functional import softmax


class CausalSelfAttention(nn.Module):
    """
    Each token looks at earlier tokens and decides which ones matter.
      Query  = "what am I looking for?"
      Key    = "what do I contain?"
      Value  = "what info do I hand over if chosen?"
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.h = num_heads
        self.d = dim // num_heads                 # numbers per head
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.o_proj = Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """In: x (B, T, dim); cos/sin (T, head_dim)   Out: (B, T, dim)"""
        B, T, C = x.shape
        # 1) make q, k, v and split into heads: (B, T, C) -> (B, H, T, D)
        q = self.q_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        # 2) tell q and k where each token sits in the sentence
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # 3) similarity of every query with every key: (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        # 4) causal mask: a token may NOT peek at future tokens
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.float().masked_fill(mask, float("-inf"))
        # 5) scores -> probabilities ("attention weights"), each row sums to 1
        weights = softmax(scores, dim=-1).to(v.dtype)
        # 6) weighted mix of the values: (B, H, T, D)
        out = weights @ v
        # 7) glue heads back together and mix them: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
