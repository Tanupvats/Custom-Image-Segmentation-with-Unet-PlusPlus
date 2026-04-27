
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from car_seg.utils.config import load_config, merge_overrides
from car_seg.utils.checkpoint import load_checkpoint
from car_seg.utils.viz import make_colormap, overlay_mask, render_legend


_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _gather(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in _VALID_EXTS)
    raise FileNotFoundError(input_path)


def _parse_points(arg: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not arg:
        return None
    pts = []
    for chunk in arg.split(";"):
        x, y = chunk.split(",")
        pts.append([float(x), float(y)])
    coords = np.array(pts, dtype=np.float32)
    labels = np.ones(len(pts), dtype=np.float32)
    return coords, labels


def _parse_box(arg: str | None) -> np.ndarray | None:
    if not arg:
        return None
    x0, y0, x1, y1 = (float(v) for v in arg.split(","))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend alpha")
    # SAM-specific
    parser.add_argument("--sam-points", default=None,
                        help="Foreground points 'x1,y1;x2,y2;...' (in image pixels)")
    parser.add_argument("--sam-box", default=None, help="Bounding box 'x0,y0,x1,y1'")
    parser.add_argument("--sam-auto", action="store_true",
                        help="Run SAM automatic mask generation instead of prompted")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = merge_overrides(cfg, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = _gather(Path(args.input))

    if cfg.model.family == "sam":
        from car_seg.models.sam_model import SAMModel
        from car_seg.engine.sam_predictor import SAMPredictor
        model = SAMModel(cfg).to(device)
        ckpt = load_checkpoint(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        predictor = SAMPredictor(model, cfg, device=device)

        for path in inputs:
            img = cv2.imread(str(path))
            if img is None:
                continue
            if args.sam_auto:
                masks = predictor.auto_segment(img)
                # Render: random color per mask
                rng = np.random.default_rng(0)
                vis = img.copy()
                for m in masks:
                    color = rng.integers(0, 255, size=3).tolist()
                    seg = m["segmentation"]
                    overlay = vis.copy()
                    overlay[seg] = color
                    vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
                cv2.imwrite(str(out_dir / f"{path.stem}_sam_auto.png"), vis)
            else:
                prompt: dict = {}
                pts = _parse_points(args.sam_points)
                if pts is not None:
                    prompt["point_coords"] = torch.from_numpy(pts[0])
                    prompt["point_labels"] = torch.from_numpy(pts[1])
                bx = _parse_box(args.sam_box)
                if bx is not None:
                    prompt["boxes"] = torch.from_numpy(bx)
                if not prompt:
                    raise SystemExit("SAM prompted mode requires --sam-points or --sam-box")
                mask = predictor.predict(img, prompt)
                # Save mask + overlay
                cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), mask * 255)
                overlay = img.copy()
                overlay[mask.astype(bool)] = (0, 255, 0)
                blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
                cv2.imwrite(str(out_dir / f"{path.stem}_sam_pred.png"), blended)
        return

    # Semantic path
    from car_seg.models import build_model
    from car_seg.engine.predictor import SemanticPredictor

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    predictor = SemanticPredictor(model, cfg, device=device)
    palette = make_colormap(cfg.model.num_classes, seed=0)

    # Save legend once
    cv2.imwrite(str(out_dir / "_legend.png"), render_legend(palette)[..., ::-1])

    for path in inputs:
        img = cv2.imread(str(path))
        if img is None:
            continue
        pred = predictor.predict_image(img)  # HxW int
        # Save raw class map (uint16 lossless) and visual overlay
        cv2.imwrite(str(out_dir / f"{path.stem}_class.png"), pred.astype(np.uint16))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = overlay_mask(rgb, pred, palette, alpha=args.alpha)
        cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  {path.name} → predicted, {len(np.unique(pred))} classes present")


if __name__ == "__main__":
    main()
