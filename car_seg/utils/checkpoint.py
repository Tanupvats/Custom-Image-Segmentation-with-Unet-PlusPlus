
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model_sd: dict,
    optimizer_sd: dict | None = None,
    scheduler_sd: dict | None = None,
    scaler_sd: dict | None = None,
    ema_sd: dict | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model_sd,
        "epoch": epoch,
    }
    if optimizer_sd is not None:
        payload["optimizer"] = optimizer_sd
    if scheduler_sd is not None:
        payload["scheduler"] = scheduler_sd
    if scaler_sd is not None:
        payload["scaler"] = scaler_sd
    if ema_sd is not None:
        payload["ema"] = ema_sd
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if extra is not None:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
