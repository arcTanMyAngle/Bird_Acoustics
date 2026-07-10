#!/usr/bin/env python3
"""
losses.py - Class-Balanced Focal Loss for the v6 retrain.

Replaces nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1) in train_v4.py.
This is the SINGLE imbalance correction: do not also enable the WeightedRandomSampler or
pass inverse-frequency weights — stacking corrections skews the decision boundary (see the
comment at train_v4.py:457).

Two references combined:
  - Class-Balanced weighting, Cui et al. 2019 (CVPR): weight each class by the inverse of its
    "effective number" (1 - beta^n)/(1 - beta) rather than raw inverse frequency. Handles the
    scrub_jay (453) vs killdeer (1275) gap more gracefully than 1/n.
  - Focal term, Lin et al. 2017 (RetinaNet): (1 - p_t)^gamma down-weights easy, confident
    examples so gradient focuses on the hard owl positives and the scrub_jay confusions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(samples_per_class, beta: float = 0.999) -> torch.Tensor:
    """Cui et al. effective-number weights, normalized so sum == n_classes
    (keeps the loss scale comparable to unweighted CE)."""
    n = torch.as_tensor(samples_per_class, dtype=torch.float)
    effective = 1.0 - torch.pow(beta, n)
    w = (1.0 - beta) / effective.clamp_min(1e-12)
    return w / w.sum() * len(n)


class CBFocalLoss(nn.Module):
    """Class-balanced focal loss with optional label smoothing on the CE term.

    Args:
        samples_per_class: list/tensor of per-class TRAIN counts (order == class index).
        gamma: focal focusing parameter (0 -> plain weighted CE; 2.0 is the standard default).
        beta:  class-balance strength; 0.999 for this dataset (~0.9999 for very large sets).
        label_smoothing: smoothing applied to the CE term only (kept small so the softmax
                         rejection threshold still has headroom).
    """

    def __init__(self, samples_per_class, gamma: float = 2.0, beta: float = 0.999,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("alpha", class_balanced_weights(samples_per_class, beta))
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)                      # (B, C)
        if self.ls > 0:
            c = logits.size(1)
            smooth = F.one_hot(target, c).float() * (1 - self.ls) + self.ls / c
            ce = -(smooth * logp).sum(1)                         # (B,)
        else:
            ce = F.nll_loss(logp, target, reduction="none")
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()  # prob of true class
        focal = (1.0 - pt).pow(self.gamma) * ce
        return (self.alpha.to(logits.device)[target] * focal).mean()


if __name__ == "__main__":
    # Sanity check against the v6-cleaned augmented counts (pre-rebalance placeholder).
    counts = [867, 900, 783, 453, 735, 1275, 960, 885, 1143]
    crit = CBFocalLoss(counts, gamma=2.0)
    w = crit.alpha
    names = ["crow", "bg", "quail", "jay", "owl", "killdeer", "dove", "hawk", "meadowlark"]
    print("class-balanced weights (sum == n_classes):")
    for name, wi, c in zip(names, w.tolist(), counts):
        print(f"  {name:12s} n={c:5d}  w={wi:.3f}")
    logits = torch.randn(8, 9, requires_grad=True)
    target = torch.randint(0, 9, (8,))
    loss = crit(logits, target)
    loss.backward()
    print(f"\nloss={loss.item():.4f}  grad_ok={logits.grad is not None}")
