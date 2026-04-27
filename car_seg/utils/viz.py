
from __future__ import annotations

import colorsys
from typing import Sequence

import numpy as np


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(image: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """image: HWC float in normalized space → HWC uint8 RGB."""
    x = image.astype(np.float32)
    x = x * std + mean
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def make_colormap(num_classes: int, seed: int = 0) -> np.ndarray:
    """Deterministic distinct colormap for `num_classes` (HSV-spread, then shuffled).

    Returns: [num_classes, 3] uint8 array. Class 0 is forced to black if you want
    background to be neutral — uncomment that line if you need it.
    """
    colors = []
    for i in range(num_classes):
        h = (i / max(num_classes, 1)) % 1.0
        # Vary saturation and value slightly so adjacent classes look different
        s = 0.55 + 0.30 * ((i * 7) % 5) / 4
        v = 0.65 + 0.30 * ((i * 13) % 3) / 2
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    rng = np.random.default_rng(seed)
    colors = np.array(colors, dtype=np.uint8)
    rng.shuffle(colors)
    # If you want class 0 = background (black), set: colors[0] = [0, 0, 0]
    return colors


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """mask: HW int → HW3 uint8."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    out[valid] = palette[mask[valid]]
    return out


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """image: HW3 uint8, mask: HW int, palette: [C,3] uint8 → HW3 uint8 blended."""
    color_mask = colorize_mask(mask, palette)
    blended = (image.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_legend(
    palette: np.ndarray,
    class_names: Sequence[str] | None = None,
    width: int = 200,
    swatch_h: int = 18,
) -> np.ndarray:
    """Render a vertical color legend as a HWC uint8 image."""
    n = len(palette)
    if class_names is None:
        class_names = [f"class {i}" for i in range(n)]
    height = swatch_h * n
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    try:
        import cv2  # local import — viz is optional
    except ImportError:
        return img
    for i, color in enumerate(palette):
        y = i * swatch_h
        img[y : y + swatch_h, 0:swatch_h] = color
        cv2.putText(
            img,
            f"{i}: {class_names[i]}",
            (swatch_h + 4, y + swatch_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return img
