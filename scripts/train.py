
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from car_seg.utils.config import load_config, merge_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Optional key=value overrides, e.g. train.lr=1e-4 model.num_classes=10",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = merge_overrides(cfg, args.override)

    if cfg.model.family == "sam":
        from car_seg.engine.sam_trainer import train_sam
        train_sam(cfg, resume=args.resume)
    else:
        from car_seg.engine.trainer import train_semantic
        train_semantic(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
