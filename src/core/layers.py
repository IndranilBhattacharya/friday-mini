"""Building blocks: Linear, Embedding, RMSNorm, RoPE. All hand-written."""
import torch
import torch.nn as nn


class Linear(nn.Module):
    """y = x @ W^T.  The basic 'neuron layer': every output is a weighted sum of all inputs."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # W shape (out, in). Start with small random numbers (std 0.02).
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In: (..., in_dim)  Out: (..., out_dim)"""
        return x @ self.weight.t()


class Embedding(nn.Module):
    """Lookup table: token id -> vector of `dim` numbers (the token's 'meaning')."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """In: (B, T) int ids   Out: (B, T, dim)"""
        return self.weight[ids]          # just pick row number `id` from the table


class RMSNorm(nn.Module):
    """
    Keeps numbers at a steady size so training stays stable.
    y = x / sqrt(mean(x^2) + eps) * g      (g = learned per-channel scale)
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In: (..., dim)  Out: (..., dim)"""
        x32 = x.float()                                           # do the math in high precision
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)   # (..., 1)
        return (x32 * rms).to(x.dtype) * self.weight


class Dropout(nn.Module):
    """Randomly zeroes elements with probability p during training. Scales by 1/(1-p)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In: (..., dim)  Out: (..., dim)"""
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return (x * mask) / (1.0 - self.p)


def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device=None):
    """
    RoPE = Rotary Position Embedding. Gives the model a sense of word ORDER by
    rotating each (q, k) vector by an angle that grows with the position.
    Out: cos, sin each of shape (seq_len, head_dim)
    """
    # One rotation speed per pair of numbers: fast for early pairs, slow for later ones.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))  # (D/2,)
    pos = torch.arange(seq_len, device=device).float()                                              # (T,)
    angles = pos[:, None] * inv_freq[None, :]                                                       # (T, D/2)
    angles = torch.cat([angles, angles], dim=-1)                                                    # (T, D)
    return torch.cos(angles), torch.sin(angles)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in two halves (a, b) and return (-b, a). This is a 90-degree turn."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate vectors: x*cos + rotate_half(x)*sin  (the 2-D rotation formula).
    In : x (B, H, T, D); cos, sin (T, D)     Out: (B, H, T, D)
    """
    return x * cos + _rotate_half(x) * sin
