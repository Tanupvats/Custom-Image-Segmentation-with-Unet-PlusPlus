
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from car_seg.models.sam_model import SAMModel


_SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class SAMPredictor:
    def __init__(self, model: SAMModel, cfg, device: str | torch.device = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device).eval()
        self.image_size = cfg.data.image_size

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]
        rgb_resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        x = rgb_resized.astype(np.float32)
        x = (x - _SAM_PIXEL_MEAN) / _SAM_PIXEL_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        return x, (H0, W0)

    def _scale_prompt(
        self, prompt: dict, orig_size: tuple[int, int]
    ) -> dict:
        """Scale user-provided prompt coordinates from original image space to
        the model input space (image_size×image_size)."""
        H0, W0 = orig_size
        sx = self.image_size / W0
        sy = self.image_size / H0
        out: dict = {}
        if "point_coords" in prompt:
            pc = prompt["point_coords"].float().clone()
            pc[..., 0] *= sx
            pc[..., 1] *= sy
            out["point_coords"] = pc.to(self.device)
            out["point_labels"] = prompt["point_labels"].float().to(self.device)
        if "boxes" in prompt:
            b = prompt["boxes"].float().clone()
            b[..., 0] *= sx
            b[..., 2] *= sx
            b[..., 1] *= sy
            b[..., 3] *= sy
            out["boxes"] = b.to(self.device)
        return out

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, prompt: dict) -> np.ndarray:
        """Predict a single binary mask given a prompt.

        prompt keys (any combination):
            point_coords: Tensor[N,2] in original-image pixel coords (xy)
            point_labels: Tensor[N]   1=foreground, 0=background
            boxes:        Tensor[K,4] in original-image pixel coords (x0y0x1y1)

        Returns: HxW uint8 binary mask at original resolution.
        """
        x, (H0, W0) = self._preprocess(image_bgr)
        scaled = self._scale_prompt(prompt, (H0, W0))
        out = self.model(x, [scaled])
        logits = out["logits"][:, 0]  # [1,H,W] at image_size
        # Resize back
        mask_at_size = (logits > 0).float()
        mask_resized = torch.nn.functional.interpolate(
            mask_at_size.unsqueeze(0), size=(H0, W0), mode="nearest"
        ).squeeze().cpu().numpy().astype(np.uint8)
        return mask_resized

    @torch.no_grad()
    def auto_segment(
        self,
        image_bgr: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.85,
        stability_score_thresh: float = 0.90,
    ) -> list[dict]:
        """Class-agnostic auto segmentation using a uniform point grid.

        Returns: list of dicts with keys 'segmentation' (HxW bool), 'iou_pred'.
        For class-aware masks, post-process each by majority-voting against a
        semantic model's prediction map.
        """
        H0, W0 = image_bgr.shape[:2]
        x, _ = self._preprocess(image_bgr)
        # Uniform point grid in input coords
        ys = np.linspace(0, self.image_size - 1, points_per_side)
        xs = np.linspace(0, self.image_size - 1, points_per_side)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        grid = np.stack([gx.flatten(), gy.flatten()], axis=-1).astype(np.float32)

        out_masks = []
        # Process in chunks to avoid OOM
        chunk = 32
        for i in range(0, len(grid), chunk):
            pts = grid[i : i + chunk]  # [c, 2]
            prompts = []
            for p in pts:
                prompts.append({
                    "point_coords": torch.from_numpy(p).unsqueeze(0).float().to(self.device),
                    "point_labels": torch.tensor([1.0], device=self.device),
                })
            # We need to encode the same image once and run the decoder per prompt.
            # The wrapper will re-encode each time inefficiently; for speed,
            # compute embeddings once and call mask decoder ourselves.
            with torch.no_grad():
                emb = self.model.sam.image_encoder(x.expand(1, -1, -1, -1))
            for p in prompts:
                pc = p["point_coords"].unsqueeze(0)
                pl = p["point_labels"].unsqueeze(0)
                sparse, dense = self.model.sam.prompt_encoder(points=(pc, pl), boxes=None, masks=None)
                low_res, iou_pred = self.model.sam.mask_decoder(
                    image_embeddings=emb,
                    image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=True,
                )
                # Pick best of the 3 by predicted IoU
                best = iou_pred.argmax(dim=1).item()
                logits = torch.nn.functional.interpolate(
                    low_res, size=(H0, W0), mode="bilinear", align_corners=False
                )[0, best]
                m = (logits > 0).cpu().numpy()
                score = float(iou_pred[0, best].item())
                if score < pred_iou_thresh:
                    continue
                # Stability score: how stable is the mask if we shift the threshold?
                hi = (logits > 1.0).float().sum().item()
                lo = (logits > -1.0).float().sum().item()
                if lo == 0:
                    continue
                stability = hi / max(lo, 1e-6)
                if stability < stability_score_thresh:
                    continue
                out_masks.append({"segmentation": m, "iou_pred": score, "stability": stability})
        return out_masks
