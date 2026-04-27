
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation


class SegFormerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # `id2label` is just metadata for HF; the head's output dim follows num_labels.
        id2label = {i: f"class_{i}" for i in range(cfg.model.num_classes)}
        label2id = {v: k for k, v in id2label.items()}

        self.net = SegformerForSemanticSegmentation.from_pretrained(
            cfg.model.hf_id,
            num_labels=cfg.model.num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=cfg.model.get("ignore_mismatched_sizes", True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF expects pixel_values; output.logits is [B, C, H/4, W/4]
        out = self.net(pixel_values=x)
        logits = out.logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
