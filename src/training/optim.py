"""AdamW optimizer + learning-rate schedule, hand-written."""
import math
import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW = "smart gradient descent". For every weight it remembers:
      m = running average of the gradient          (which direction is it pushing?)
      v = running average of the gradient squared  (how bumpy/noisy is it?)
    Update: weight -= lr * m / (sqrt(v) + eps)  and a tiny pull towards zero (weight decay).
    (We inherit torch.optim.Optimizer only as a container; the math below is ours.
     Inheriting also lets the fp16 GradScaler talk to us.)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:                                   # first time: create memory
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                st["m"].mul_(b1).add_(g, alpha=1 - b1)                # m = b1*m + (1-b1)*g
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)         # v = b2*v + (1-b2)*g*g
                m_hat = st["m"] / (1 - b1 ** t)                       # fix the early-steps bias
                v_hat = st["v"] / (1 - b2 ** t)
                p.mul_(1 - group["lr"] * group["weight_decay"])       # weight decay (pull to 0)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])


def lr_at(step: int, total_steps: int, peak_lr: float, warmup: int, min_ratio: float) -> float:
    """
    Learning rate for this step: ramp up linearly, then glide down on a cosine curve.
    (Big steps early to learn fast, tiny steps late to settle.)
    """
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)          # 0 -> 1
    cosine = 0.5 * (1 + math.cos(math.pi * progress))                  # 1 -> 0
    return peak_lr * (min_ratio + (1 - min_ratio) * cosine)
