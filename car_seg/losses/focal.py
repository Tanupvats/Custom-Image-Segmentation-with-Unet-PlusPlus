
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        mode: str = "multiclass",
        ignore_index: int | None = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        if mode not in {"binary", "multiclass"}:
            raise ValueError(f"Unknown Focal mode: {mode}")
        self.alpha = alpha
        self.gamma = gamma
        self.mode = mode
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "binary":
            return self._binary(logits, targets)
        return self._multiclass(logits, targets)

    def _binary(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        targets_f = targets.float()
        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
        else:
            valid = torch.ones_like(targets, dtype=torch.bool)

        bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets_f + (1 - p) * (1 - targets_f)
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce

        loss = loss * valid.float()
        if self.reduction == "mean":
            denom = valid.float().sum().clamp_min(1.0)
            return loss.sum() / denom
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _multiclass(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits [B,C,H,W], targets [B,H,W]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.size(1)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            targets_clamped = targets.clone()
            targets_clamped[~valid] = 0
        else:
            valid = torch.ones_like(targets, dtype=torch.bool)
            targets_clamped = targets

        targets_oh = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
        ce = -(targets_oh * log_probs).sum(dim=1)         # [B,H,W]
        p_t = (targets_oh * probs).sum(dim=1)             # [B,H,W]
        focal_term = (1 - p_t).pow(self.gamma)
        loss = self.alpha * focal_term * ce

        loss = loss * valid.float()
        if self.reduction == "mean":
            denom = valid.float().sum().clamp_min(1.0)
            return loss.sum() / denom
        if self.reduction == "sum":
            return loss.sum()
        return loss
