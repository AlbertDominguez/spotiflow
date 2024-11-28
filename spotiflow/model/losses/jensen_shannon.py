"""
Adapted from kornia: https://github.com/kornia/kornia/blob/main/kornia/losses/divergence.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kl_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    From Kornia: https://github.com/kornia/kornia/blob/main/kornia/losses/divergence.py
    """
    # D_KL(P || Q)
    batch, chans, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch * chans, height * width).log(), p.reshape(batch * chans, height * width), reduction="none"
    )
    kl_values = unsummed_kl.view(batch, chans, height, width)
    return kl_values


def _js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    From Kornia: https://github.com/kornia/kornia/blob/main/kornia/losses/divergence.py
    """
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)

def _kl_div_3d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # D_KL(P || Q)
    batch, chans, depth, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch * chans, depth * height * width).log(), p.reshape(batch * chans, depth * height * width), reduction="none"
    )
    kl_values = unsummed_kl.view(batch, chans, depth, height, width)
    return kl_values

def _js_div_3d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_3d(p, m) + 0.5 * _kl_div_3d(q, m)

class JensenShannonDivergence(nn.Module):
    def __init__(self, eps=1e-20, reduction="mean"):
        super().__init__()
        self._reduction = reduction
        self._eps = eps
    
    def forward(self, input, target):
        if input.ndim == 4:
            loss = _js_div_2d(torch.clamp(input, self._eps, 1-self._eps), torch.clamp(target, self._eps, 1-self._eps))
        else:
            loss = _js_div_3d(torch.clamp(input, self._eps, 1-self._eps), torch.clamp(target, self._eps, 1-self._eps))
        if self._reduction == "none":
            return loss
        elif self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction {self._reduction}")


if __name__ == "__main__":
    inp = torch.randn(3, 1,64,512, 512)
    target = torch.randn(3, 1,64, 512, 512)
    loss = JensenShannonDivergence(reduction="mean")
    print(loss(inp, target))
