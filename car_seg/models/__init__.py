"""Model factory."""
from __future__ import annotations


def build_model(cfg):
    family = cfg.model.family
    if family == "unetpp":
        from .unetpp import UNetPlusPlusModel
        return UNetPlusPlusModel(cfg)
    if family == "segformer":
        from .segformer import SegFormerModel
        return SegFormerModel(cfg)
    if family == "sam":
        from .sam_model import SAMModel
        return SAMModel(cfg)
    raise ValueError(f"Unknown model family: {family}")
