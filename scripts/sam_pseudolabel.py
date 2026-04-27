
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from car_seg.utils.config import load_config
from car_seg.utils.checkpoint import load_checkpoint


def _refine_with_sam(
    image_bgr: np.ndarray,
    coarse_mask: np.ndarray,
    sam_predictor,
    num_classes: int,
    min_component_pixels: int = 200,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    refined = np.full_like(coarse_mask, fill_value=255)  # ignore by default
    for cls in range(num_classes):
        bin_mask = (coarse_mask == cls).astype(np.uint8)
        if bin_mask.sum() < min_component_pixels:
            continue
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=4)
        for c in range(1, n):
            area = stats[c, cv2.CC_STAT_AREA]
            if area < min_component_pixels:
                continue
            cx, cy = centroids[c]
            prompt = {
                "point_coords": torch.tensor([[cx, cy]], dtype=torch.float32),
                "point_labels": torch.tensor([1.0], dtype=torch.float32),
            }
            sam_mask = sam_predictor.predict(image_bgr, prompt)
            # IoU between SAM mask and the coarse component
            comp = (labels == c).astype(np.uint8)
            inter = (sam_mask & comp).sum()
            union = (sam_mask | comp).sum()
            iou = inter / max(union, 1)
            if iou < iou_threshold:
                continue
            refined[sam_mask.astype(bool)] = cls
    return refined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-config", required=True, help="Config of the coarse model")
    parser.add_argument("--semantic-ckpt", required=True)
    parser.add_argument("--sam-config", required=True)
    parser.add_argument("--sam-ckpt", required=True)
    parser.add_argument("--input", required=True, help="Directory of unlabeled images")
    parser.add_argument("--output", required=True, help="Directory to write pseudo-masks")
    parser.add_argument("--min-pixels", type=int, default=200)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    sem_cfg = load_config(args.semantic_config)
    sam_cfg = load_config(args.sam_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from car_seg.models import build_model
    from car_seg.engine.predictor import SemanticPredictor
    from car_seg.models.sam_model import SAMModel
    from car_seg.engine.sam_predictor import SAMPredictor

    sem_model = build_model(sem_cfg).to(device).eval()
    sem_model.load_state_dict(load_checkpoint(args.semantic_ckpt, map_location=device)["model"])
    sem_predictor = SemanticPredictor(sem_model, sem_cfg, device=device)

    sam_model = SAMModel(sam_cfg).to(device).eval()
    sam_model.load_state_dict(load_checkpoint(args.sam_ckpt, map_location=device)["model"])
    sam_predictor = SAMPredictor(sam_model, sam_cfg, device=device)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in valid_exts)
    print(f"Pseudo-labeling {len(paths)} images...")

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        coarse = sem_predictor.predict_image(img)  # HxW int
        refined = _refine_with_sam(
            img, coarse, sam_predictor,
            num_classes=sem_cfg.model.num_classes,
            min_component_pixels=args.min_pixels,
            iou_threshold=args.iou_threshold,
        )
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), refined.astype(np.uint8))
        print(f"  {path.name} done")


if __name__ == "__main__":
    main()
