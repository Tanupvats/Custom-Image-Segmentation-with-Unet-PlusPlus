
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from car_seg.data.transforms import build_val_transform


class SemanticPredictor:
    def __init__(self, model: torch.nn.Module, cfg, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device).eval()
        self.transform = build_val_transform(cfg)
        self.image_size = cfg.data.image_size

    @torch.no_grad()
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(images)
        if isinstance(logits, dict):
            logits = logits["main"]
        return logits

    @torch.no_grad()
    def _forward_with_tta(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward(images)
        # Horizontal flip TTA
        logits_h = self._forward(torch.flip(images, dims=[-1]))
        logits_h = torch.flip(logits_h, dims=[-1])
        return (logits.softmax(dim=1) + logits_h.softmax(dim=1)) / 2

    @torch.no_grad()
    def _sliding_window(self, image_tensor: torch.Tensor, window: int, stride: int) -> torch.Tensor:
        """image_tensor: [1,3,H,W] already normalized. Returns [1,C,H,W] probs."""
        _, _, H, W = image_tensor.shape
        num_classes = self.cfg.model.num_classes
        accum = torch.zeros((1, num_classes, H, W), device=image_tensor.device)
        count = torch.zeros((1, 1, H, W), device=image_tensor.device)
        for y in range(0, max(H - window, 0) + 1, stride):
            for x in range(0, max(W - window, 0) + 1, stride):
                y2 = min(y + window, H)
                x2 = min(x + window, W)
                y1 = y2 - window
                x1 = x2 - window
                tile = image_tensor[:, :, y1:y2, x1:x2]
                tile_logits = self._forward_with_tta(tile) if self.cfg.eval.get("tta") else \
                    self._forward(tile).softmax(dim=1)
                accum[:, :, y1:y2, x1:x2] += tile_logits
                count[:, :, y1:y2, x1:x2] += 1.0
        return accum / count.clamp_min(1.0)

    def predict_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Predict a class map for a single image.

        Args:
            image_bgr: HxWx3 uint8 BGR (as cv2.imread returns).
        Returns:
            HxW int32 class id map at the original resolution.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]

        if self.cfg.eval.get("sliding_window") and (H0 > self.image_size or W0 > self.image_size):
            # Normalize at original resolution; transform.Resize is skipped here
            # because we tile.
            from car_seg.data.transforms import _IMAGENET_MEAN, _IMAGENET_STD
            x = rgb.astype(np.float32) / 255.0
            x = (x - np.array(_IMAGENET_MEAN)) / np.array(_IMAGENET_STD)
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
            window = int(self.cfg.eval.sliding_window_size)
            stride = int(self.cfg.eval.sliding_window_stride)
            # Pad so window fits
            pad_h = max(0, window - H0)
            pad_w = max(0, window - W0)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            probs = self._sliding_window(x, window=window, stride=stride)
            # Crop padding
            probs = probs[:, :, :H0, :W0]
            preds = probs.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
            return preds

        # Standard path: resize → forward → upsample to original
        aug = self.transform(image=rgb, mask=np.zeros(rgb.shape[:2], np.uint8))
        x = aug["image"].unsqueeze(0).to(self.device)
        if self.cfg.eval.get("tta"):
            probs = self._forward_with_tta(x)
        else:
            probs = self._forward(x).softmax(dim=1)
        probs = F.interpolate(probs, size=(H0, W0), mode="bilinear", align_corners=False)
        preds = probs.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        return preds

    def predict_paths(self, paths: Iterable[str | Path]) -> list[tuple[str, np.ndarray]]:
        out = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            pred = self.predict_image(img)
            out.append((str(p), pred))
        return out
