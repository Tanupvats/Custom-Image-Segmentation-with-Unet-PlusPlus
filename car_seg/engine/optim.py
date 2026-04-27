
from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


def build_optimizer(model, cfg):
    name = cfg.train.optimizer.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.train.lr,
            betas=tuple(cfg.train.get("betas", [0.9, 0.999])),
            weight_decay=cfg.train.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.train.lr,
            betas=tuple(cfg.train.get("betas", [0.9, 0.999])),
            weight_decay=cfg.train.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.train.lr,
            momentum=cfg.train.get("momentum", 0.9),
            weight_decay=cfg.train.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {name}")


class WarmupWrapper(_LRScheduler):
    """Linear warmup from 0 → base_lr over `warmup_steps`, then defers to inner."""

    def __init__(self, optimizer, inner, warmup_steps: int, last_epoch: int = -1):
        self.inner = inner
        self.warmup_steps = max(int(warmup_steps), 0)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps and self.warmup_steps > 0:
            scale = (step + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        # Hand off to inner
        return [g["lr"] for g in self.optimizer.param_groups]

    def step(self, epoch=None):
        # During warmup, manually set LRs; afterwards step the inner scheduler.
        self.last_epoch += 1
        step = self.last_epoch
        if step < self.warmup_steps and self.warmup_steps > 0:
            scale = (step + 1) / self.warmup_steps
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base_lr * scale
        else:
            # Once warmup is done, the inner scheduler takes over the actual stepping.
            self.inner.step()


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    epochs = cfg.train.epochs
    sched_name = cfg.train.scheduler.lower()
    warmup_epochs = cfg.train.get("warmup_epochs", 0)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    if sched_name == "none":
        inner = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    elif sched_name == "cosine":
        # Cosine over the post-warmup window
        post_steps = max(total_steps - warmup_steps, 1)
        inner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=post_steps)
    elif sched_name == "poly":
        power = cfg.train.get("poly_power", 1.0)

        def poly_fn(step: int) -> float:
            return max(0.0, (1.0 - step / max(total_steps - warmup_steps, 1))) ** power

        inner = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_fn)
    elif sched_name == "plateau":
        inner = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4, threshold=1e-3
        )
        # Plateau is stepped manually in trainer with the val metric
        return inner
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

    if warmup_steps > 0 and sched_name != "plateau":
        return WarmupWrapper(optimizer, inner, warmup_steps=warmup_steps)
    return inner
