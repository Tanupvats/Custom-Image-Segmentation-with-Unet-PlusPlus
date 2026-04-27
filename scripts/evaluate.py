
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from car_seg.utils.config import load_config, merge_overrides
from car_seg.utils.checkpoint import load_checkpoint
from car_seg.utils.logging import setup_stdout_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    log = setup_stdout_logger("evaluate")
    cfg = load_config(args.config)
    if args.override:
        cfg = merge_overrides(cfg, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.family == "sam":
        from car_seg.engine.sam_trainer import evaluate_sam
        from car_seg.data.sam_dataset import build_sam_loader
        from car_seg.models.sam_model import SAMModel

        model = SAMModel(cfg).to(device)
        ckpt = load_checkpoint(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        val_loader = build_sam_loader(cfg, train=False)
        miou = evaluate_sam(model, val_loader, device, use_amp=False)
        log.info(f"SAM val mIoU (binary, prompted): {miou:.4f}")
        return

    from car_seg.engine.trainer import evaluate_semantic
    from car_seg.data import build_val_loader
    from car_seg.data.transforms import build_val_transform
    from car_seg.models import build_model

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_loader = build_val_loader(cfg, build_val_transform(cfg))

    metrics = evaluate_semantic(model, val_loader, device, cfg, use_amp=False)
    print(metrics.format_table())
    log.info(f"Summary: {metrics.to_summary()}")


if __name__ == "__main__":
    main()
