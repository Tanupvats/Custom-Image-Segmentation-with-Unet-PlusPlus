
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_stdout_logger(name: str = "car_seg", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class MetricLogger:
    """Wraps TensorBoard + W&B with a uniform `log(dict, step)` interface."""

    def __init__(
        self,
        output_dir: str | Path,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tb = None
        self._wandb = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            except ImportError:
                pass

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=wandb_project or "car-parts-seg",
                    name=run_name,
                    dir=str(self.output_dir),
                    config=config or {},
                )
            except ImportError:
                pass

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_image(self, tag: str, image, step: int) -> None:
        """image: HWC uint8 numpy array."""
        if self._tb is not None:
            # TB expects CHW
            self._tb.add_image(tag, image.transpose(2, 0, 1), step)
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Image(image)}, step=step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
