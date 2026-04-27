
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class SegmentationMetrics:
    """Streaming multi-class segmentation metrics.

    Args:
        num_classes: number of foreground classes
        ignore_index: pixels equal to this label are excluded from metrics
    """

    def __init__(self, num_classes: int, ignore_index: int | None = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions.

        preds:   [N, H, W] integer class predictions, OR [N, C, H, W] logits/probs
        targets: [N, H, W] integer ground truth class ids
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        assert preds.shape == targets.shape, (preds.shape, targets.shape)

        preds_np = preds.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)

        if self.ignore_index is not None:
            valid = targets_np != self.ignore_index
            preds_np = preds_np[valid]
            targets_np = targets_np[valid]

        # Also drop any prediction that's outside [0, num_classes) — shouldn't
        # happen but keeps things robust.
        valid = (targets_np >= 0) & (targets_np < self.num_classes)
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        # Bincount-based confusion matrix update — much faster than nested loops
        index = self.num_classes * targets_np.astype(np.int64) + preds_np.astype(np.int64)
        bincount = np.bincount(index, minlength=self.num_classes**2)
        self._cm += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self) -> "MetricsResult":
        cm = self._cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Per-class IoU. Classes never seen in GT and never predicted produce NaN
        # so we can detect "not present" vs "predicted but wrong".
        denom = tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            iou_per_class = np.where(denom > 0, tp / denom, np.nan)

        # mIoU = mean over classes that actually appeared (in GT or pred)
        miou = float(np.nanmean(iou_per_class)) if np.any(~np.isnan(iou_per_class)) else 0.0

        # Frequency-weighted IoU
        freq = cm.sum(axis=1) / max(cm.sum(), 1.0)
        with np.errstate(invalid="ignore"):
            fwiou = float(np.nansum(freq * iou_per_class))

        # Pixel accuracy
        pix_acc = float(tp.sum() / max(cm.sum(), 1.0))

        # Per-class precision/recall (Dice = F1) for completeness
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
            recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
            dice = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), np.nan)

        return MetricsResult(
            miou=miou,
            fwiou=fwiou,
            pixel_acc=pix_acc,
            iou_per_class=iou_per_class,
            precision_per_class=precision,
            recall_per_class=recall,
            dice_per_class=dice,
            mean_dice=float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0,
            confusion_matrix=cm,
        )


@dataclass
class MetricsResult:
    miou: float
    fwiou: float
    pixel_acc: float
    mean_dice: float
    iou_per_class: np.ndarray
    precision_per_class: np.ndarray
    recall_per_class: np.ndarray
    dice_per_class: np.ndarray
    confusion_matrix: np.ndarray

    def to_summary(self) -> dict:
        return {
            "miou": self.miou,
            "fwiou": self.fwiou,
            "pixel_acc": self.pixel_acc,
            "mean_dice": self.mean_dice,
        }

    def format_table(self, class_names: list[str] | None = None, top_k: int | None = None) -> str:
        n = len(self.iou_per_class)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(n)]
        rows = list(zip(class_names, self.iou_per_class, self.dice_per_class))
        # Sort by IoU desc, NaN last
        rows.sort(key=lambda r: (-(r[1] if not np.isnan(r[1]) else -1.0)))
        if top_k:
            rows = rows[:top_k]
        lines = [f"{'class':<24} {'IoU':>8} {'Dice':>8}"]
        for name, iou, dice in rows:
            iou_s = f"{iou:.4f}" if not np.isnan(iou) else "  n/a"
            d_s = f"{dice:.4f}" if not np.isnan(dice) else "  n/a"
            lines.append(f"{name:<24} {iou_s:>8} {d_s:>8}")
        lines.append("-" * 42)
        lines.append(f"{'mIoU':<24} {self.miou:>8.4f}")
        lines.append(f"{'fwIoU':<24} {self.fwiou:>8.4f}")
        lines.append(f"{'pixel acc':<24} {self.pixel_acc:>8.4f}")
        lines.append(f"{'mean Dice':<24} {self.mean_dice:>8.4f}")
        return "\n".join(lines)


@torch.no_grad()
def binary_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-7) -> float:
    """IoU between two binary masks. Used for SAM per-prompt evaluation."""
    pred = pred_mask.bool()
    gt = gt_mask.bool()
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    if union == 0:
        return float("nan")
    return inter / (union + eps)
