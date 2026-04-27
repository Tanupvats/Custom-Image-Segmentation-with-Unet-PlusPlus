
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from car_seg.utils.config import load_config
from car_seg.utils.checkpoint import load_checkpoint
from car_seg.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true",
                        help="Allow dynamic batch and spatial dims")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.family == "sam":
        raise SystemExit("Use Meta's official SAM exporter for SAM models.")

    device = torch.device("cpu")  # export on CPU for portability
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    h = w = cfg.data.image_size
    dummy = torch.randn(1, 3, h, w, device=device)
    output = args.output or str(Path(args.ckpt).with_suffix(".onnx"))

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        }

    # Wrap to ensure dict outputs collapse to a tensor
    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            y = self.m(x)
            if isinstance(y, dict):
                y = y["main"]
            return y

    torch.onnx.export(
        _Wrap(model).eval(),
        dummy,
        output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"Exported ONNX to {output}")

    # Sanity check: load with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})
        print(f"ORT sanity check OK. Output shape: {out[0].shape}")
    except ImportError:
        print("(onnxruntime not installed — skipping sanity check)")


if __name__ == "__main__":
    main()
