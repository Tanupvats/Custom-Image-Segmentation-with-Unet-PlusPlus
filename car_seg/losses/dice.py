
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        mode: str = "multiclass",
        from_logits: bool = True,
        smooth: float = 1.0,
        ignore_index: int | None = 255,
    ):
        super().__init__()
        if mode not in {"binary", "multiclass"}:
            raise ValueError(f"Unknown Dice mode: {mode}")
        self.mode = mode
        self.from_logits = from_logits
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "binary":
            return self._binary(logits, targets)
        return self._multiclass(logits, targets)

    def _binary(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 1, H, W] or [B, H, W]
        # targets: [B, H, W] in {0, 1} (and possibly ignore_index)
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        probs = torch.sigmoid(logits) if self.from_logits else logits

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index).float()
        else:
            valid = torch.ones_like(probs)

        targets_f = (targets.float() * valid).clamp(0, 1)
        probs = probs * valid

        dims = (1, 2)
        inter = (probs * targets_f).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets_f.sum(dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

    def _multiclass(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], targets: [B, H, W]
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1) if self.from_logits else logits

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            targets_clamped = targets.clone()
            targets_clamped[~valid] = 0  # placeholder, masked out below
        else:
            valid = torch.ones_like(targets, dtype=torch.bool)
            targets_clamped = targets

        targets_oh = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        valid_f = valid.float().unsqueeze(1)  # [B,1,H,W]

        probs = probs * valid_f
        targets_oh = targets_oh * valid_f

        dims = (0, 2, 3)  # sum over batch + spatial → per-class scores
        inter = (probs * targets_oh).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice_per_class = (2 * inter + self.smooth) / (denom + self.smooth)
        # Average over classes that actually appear (denom>0); if a class is
        # totally absent in this batch, its dice is ~1 from the smoothing — we
        # want to ignore it so it doesn't dominate the loss.
        present = (targets_oh.sum(dim=dims) > 0).float()
        if present.sum() == 0:
            return logits.new_tensor(0.0)
        loss = ((1.0 - dice_per_class) * present).sum() / present.sum()
        return loss
