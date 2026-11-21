"""Basic math functions, written by hand (no torch.nn.functional)."""
import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Turns raw scores into probabilities that add up to 1.
    In : x  (..., N)  any real numbers
    Out:    (..., N)  numbers in 0..1, sum over `dim` == 1
    """
    # Subtract the biggest number first. Same answer, but exp() can't explode.
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)                              # (..., N) always positive
    return e / e.sum(dim=dim, keepdim=True)       # divide by total => sums to 1


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (a.k.a. swish) = x * sigmoid(x). A smooth "switch" used inside the MLP.
    In : (...)   Out: (...) same shape
    """
    return x * (1.0 / (1.0 + torch.exp(-x)))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    The training loss: "how surprised was the model by the true next token?"
    loss = -log(probability the model gave to the correct token), averaged.
    In : logits  (N, V) raw scores for each of V tokens, for N positions
         targets (N,)   correct token id for each position (or ignore_index to skip)
    Out: scalar tensor
    """
    logits = logits.float()
    # log-softmax = x - log(sum(exp(x))), computed in a stable way.
    m = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(logits - m).sum(dim=-1)) + m.squeeze(-1)   # (N,)
    keep = targets != ignore_index                                               # (N,) bool
    safe_targets = torch.where(keep, targets, torch.zeros_like(targets))         # avoid index -100
    correct_score = logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)       # (N,)
    nll = (log_sum_exp - correct_score) * keep                                   # (N,) 0 where ignored
    return nll.sum() / keep.sum().clamp(min=1)
