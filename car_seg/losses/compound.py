
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .dice import DiceLoss
from .focal import FocalLoss


class CompoundLoss(nn.Module):
    def __init__(self, components: Iterable[tuple[float, nn.Module, str]]):
        """components: iterable of (weight, module, name)."""
        super().__init__()
        self._modules_list = nn.ModuleList([m for _, m, _ in components])
        self._weights = [w for w, _, _ in components]
        self._names = [n for _, _, n in components]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        total = logits.new_tensor(0.0)
        breakdown: dict[str, float] = {}
        for w, m, n in zip(self._weights, self._modules_list, self._names):
            v = m(logits, targets)
            # Guard against NaN/Inf components — can happen for CE when every pixel
            # is ignore_index (degenerate batch). Replace with zero so the rest of
            # training continues; the optimizer just gets no signal from this batch.
            if not torch.isfinite(v):
                v = logits.new_tensor(0.0)
            total = total + w * v
            breakdown[n] = float(v.detach().item())
        breakdown["total"] = float(total.detach().item())
        return total, breakdown


def build_loss(cfg) -> CompoundLoss:
    components: list[tuple[float, nn.Module, str]] = []
    ignore_index = cfg.data.ignore_index
    class_weights = cfg.loss.get("class_weights")
    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32)

    for spec in cfg.loss.components:
        t = spec.type
        w = float(spec.weight)
        name = f"{t}_{len(components)}"
        if t == "dice":
            mod = DiceLoss(
                mode=spec.get("mode", "multiclass"),
                from_logits=spec.get("from_logits", True),
                smooth=spec.get("smooth", 1.0),
                ignore_index=ignore_index,
            )
        elif t == "focal":
            mod = FocalLoss(
                alpha=spec.get("alpha", 0.25),
                gamma=spec.get("gamma", 2.0),
                mode=spec.get("mode", "multiclass"),
                ignore_index=ignore_index,
            )
        elif t == "cross_entropy":
            mod = nn.CrossEntropyLoss(
                weight=cw_tensor,
                ignore_index=ignore_index,
                label_smoothing=spec.get("label_smoothing", 0.0),
            )
            # Wrap so signature matches (logits, targets) -> scalar
            mod = _CEWrapper(mod)
        elif t == "bce":
            mod = _BCEWrapper(ignore_index=ignore_index)
        else:
            raise ValueError(f"Unknown loss type: {t}")
        components.append((w, mod, name))
    return CompoundLoss(components)


class _CEWrapper(nn.Module):
    def __init__(self, ce: nn.CrossEntropyLoss):
        super().__init__()
        self.ce = ce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class _BCEWrapper(nn.Module):
    def __init__(self, ignore_index: int | None = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        targets_f = targets.float()
        if self.ignore_index is not None:
            valid = (targets != self.ignore_index).float()
        else:
            valid = torch.ones_like(targets_f)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_f.clamp(0, 1), reduction="none"
        )
        loss = loss * valid
        return loss.sum() / valid.sum().clamp_min(1.0)
