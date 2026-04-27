
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CarPartsDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform=None,
        mask_suffix_replace: tuple[str, str] | None = None,
        ignore_index: int = 255,
        num_classes: int | None = None,
        validate_masks: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_suffix_replace = tuple(mask_suffix_replace) if mask_suffix_replace else None
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        if not self.images_dir.exists():
            raise FileNotFoundError(f"images_dir does not exist: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"masks_dir does not exist: {self.masks_dir}")

        self.images_list = sorted(
            f for f in os.listdir(self.images_dir)
            if Path(f).suffix.lower() in _VALID_IMAGE_EXTS
        )
        if not self.images_list:
            raise RuntimeError(f"No images found in {self.images_dir}")

        if validate_masks:
            self._validate_first_n(n=8)

    def __len__(self) -> int:
        return len(self.images_list)

    def _mask_path_for(self, image_name: str) -> Path:
        if self.mask_suffix_replace is None:
            return self.masks_dir / image_name
        old, new = self.mask_suffix_replace
        return self.masks_dir / image_name.replace(old, new)

    def _validate_first_n(self, n: int = 8) -> None:
        """Spot-check that masks load and contain plausible class IDs."""
        for name in self.images_list[:n]:
            mp = self._mask_path_for(name)
            if not mp.exists():
                raise FileNotFoundError(
                    f"Mask not found: {mp}. "
                    f"Hint: set data.mask_suffix_replace if extensions differ."
                )
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"cv2.imread returned None for mask: {mp}")
            if self.num_classes is not None:
                vals = np.unique(m)
                bad = [int(v) for v in vals if v != self.ignore_index and v >= self.num_classes]
                if bad:
                    raise ValueError(
                        f"Mask {mp} contains class IDs >= num_classes ({self.num_classes}): {bad}. "
                        "Either increase num_classes, remap your masks, or set ignore_index."
                    )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        name = self.images_list[idx]
        img_path = self.images_dir / name
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"cv2.imread returned None for image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self._mask_path_for(name)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cv2.imread returned None for mask: {mask_path}")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # tensor CHW after ToTensorV2
            mask = augmented["mask"]     # tensor HW (uint8/int64), preserved by A
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        # CrossEntropyLoss wants long type
        mask = mask.long()

        return {"image": image, "mask": mask, "image_path": str(img_path)}


def _collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "image_path": [b["image_path"] for b in batch],
    }


def build_train_loader(cfg, transform) -> DataLoader:
    ds = CarPartsDataset(
        cfg.data.train_images,
        cfg.data.train_masks,
        transform=transform,
        mask_suffix_replace=cfg.data.get("mask_suffix_replace"),
        ignore_index=cfg.data.ignore_index,
        num_classes=cfg.model.num_classes,
        validate_masks=True,
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
        drop_last=True,
        collate_fn=_collate,
    )


def build_val_loader(cfg, transform) -> DataLoader:
    ds = CarPartsDataset(
        cfg.data.val_images,
        cfg.data.val_masks,
        transform=transform,
        mask_suffix_replace=cfg.data.get("mask_suffix_replace"),
        ignore_index=cfg.data.ignore_index,
        num_classes=cfg.model.num_classes,
        validate_masks=True,
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
        drop_last=False,
        collate_fn=_collate,
    )
