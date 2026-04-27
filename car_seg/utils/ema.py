
from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains a shadow copy of the model with EMA-updated weights.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # in train loop, after optimizer.step():
        ema.update(model)
        # at eval time, evaluate ema.module instead of model
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: str | torch.device | None = None):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)
