
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# SAM normalization (over 0..255, applied AFTER ToTensor scales to 0..1 — see below)
_SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class SAMCarPartsDataset(Dataset):
    """SAM fine-tuning dataset with per-class prompt sampling.

    Notes:
        - Images are resized to image_size (default 1024). The mask is resized
          with NEAREST to preserve class IDs.
        - Augmentation is light because SAM is robust; we do flips + small jitter.
        - Class 0 is treated as background and is not used as a target class
          unless `include_background=True`.
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: int = 1024,
        ignore_index: int = 255,
        num_classes: int = 50,
        include_background: bool = False,
        prompts_cfg: dict | None = None,
        train: bool = True,
        mask_suffix_replace: tuple[str, str] | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.include_background = include_background
        self.train = train
        self.prompts_cfg = prompts_cfg or {}
        self.mask_suffix_replace = mask_suffix_replace

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images_list = sorted(
            f for f in os.listdir(self.images_dir)
            if Path(f).suffix.lower() in valid_exts
        )
        if not self.images_list:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images_list)

    def _mask_path_for(self, image_name: str) -> Path:
        if self.mask_suffix_replace is None:
            return self.masks_dir / image_name
        old, new = self.mask_suffix_replace
        return self.masks_dir / image_name.replace(old, new)

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        name = self.images_list[idx]
        img_path = self.images_dir / name
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        m_path = self._mask_path_for(name)
        mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read {m_path}")

        # Resize to SAM's input size
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Light augmentation
        if self.train and random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()

        return image, mask, str(img_path)

    def _sample_class(self, mask: np.ndarray) -> int | None:
        unique = np.unique(mask)
        candidates = [
            int(u) for u in unique
            if u != self.ignore_index
            and 0 <= u < self.num_classes
            and (self.include_background or u != 0)
        ]
        if not candidates:
            return None
        return random.choice(candidates)

    def _sample_point_prompt(self, binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample one foreground point (with jitter)."""
        ys, xs = np.where(binary_mask > 0)
        if len(ys) == 0:
            # Should not happen — caller filters classes that exist
            return np.zeros((1, 2), np.float32), np.zeros((1,), np.float32)
        i = random.randint(0, len(ys) - 1)
        y, x = ys[i], xs[i]
        jitter = self.prompts_cfg.get("point_jitter_px", 0)
        if jitter > 0:
            x += random.randint(-jitter, jitter)
            y += random.randint(-jitter, jitter)
        x = float(np.clip(x, 0, self.image_size - 1))
        y = float(np.clip(y, 0, self.image_size - 1))
        coords = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], dtype=np.float32)  # 1 = foreground
        return coords, labels

    def _sample_box_prompt(self, binary_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(binary_mask > 0)
        if len(ys) == 0:
            return np.zeros((4,), np.float32)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        jitter = self.prompts_cfg.get("box_jitter_px", 0)
        if jitter > 0:
            x0 = max(0, x0 - random.randint(0, jitter))
            y0 = max(0, y0 - random.randint(0, jitter))
            x1 = min(self.image_size - 1, x1 + random.randint(0, jitter))
            y1 = min(self.image_size - 1, y1 + random.randint(0, jitter))
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, mask, path = self._load_pair(idx)

        cls = self._sample_class(mask)
        if cls is None:
            # Re-sample a different image — fall back to next index
            return self.__getitem__((idx + 1) % len(self))

        binary_mask = (mask == cls).astype(np.uint8)

        # Choose prompt type
        p_point = float(self.prompts_cfg.get("point_per_class_p", 0.6))
        p_box = float(self.prompts_cfg.get("box_per_class_p", 0.4))
        total = p_point + p_box
        r = random.random() * total
        prompt: dict = {}
        if r < p_point:
            coords, labels = self._sample_point_prompt(binary_mask)
            prompt["point_coords"] = torch.from_numpy(coords)
            prompt["point_labels"] = torch.from_numpy(labels)
        else:
            box = self._sample_box_prompt(binary_mask)
            prompt["boxes"] = torch.from_numpy(box)

        # Convert image to tensor and apply SAM normalization.
        # Per Meta's reference: x = (x - pixel_mean) / pixel_std  with pixels in 0..255.
        img_t = torch.from_numpy(image.astype(np.float32))
        img_t = (img_t - torch.tensor(_SAM_PIXEL_MEAN)) / torch.tensor(_SAM_PIXEL_STD)
        img_t = img_t.permute(2, 0, 1).contiguous()  # CHW

        bin_mask_t = torch.from_numpy(binary_mask).float()  # [H,W]

        return {
            "image": img_t,
            "binary_mask": bin_mask_t,
            "prompt": prompt,
            "class_id": cls,
            "image_path": path,
        }


def _sam_collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "binary_mask": torch.stack([b["binary_mask"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "class_id": torch.tensor([b["class_id"] for b in batch], dtype=torch.long),
        "image_path": [b["image_path"] for b in batch],
    }


def build_sam_loader(cfg, train: bool) -> DataLoader:
    images_dir = cfg.data.train_images if train else cfg.data.val_images
    masks_dir = cfg.data.train_masks if train else cfg.data.val_masks
    ds = SAMCarPartsDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=cfg.data.image_size,
        ignore_index=cfg.data.ignore_index,
        num_classes=cfg.model.num_classes,
        prompts_cfg=cfg.prompts.to_dict() if hasattr(cfg.prompts, "to_dict") else dict(cfg.prompts),
        train=train,
        mask_suffix_replace=cfg.data.get("mask_suffix_replace"),
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=train,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
        drop_last=train,
        collate_fn=_sam_collate,
    )
