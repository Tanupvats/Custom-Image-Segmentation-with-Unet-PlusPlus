
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlusPlusModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.deep_supervision = bool(cfg.model.deep_supervision)
        self.net = smp.UnetPlusPlus(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.model.num_classes,
            decoder_channels=tuple(cfg.model.decoder_channels),
            activation=None,  # we want raw logits
        )

        if self.deep_supervision:
            # Auxiliary head on the deepest encoder feature
            enc_out_chs = self.net.encoder.out_channels[-1]
            self.aux_head = nn.Sequential(
                nn.Conv2d(enc_out_chs, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, cfg.model.num_classes, kernel_size=1),
            )
        else:
            self.aux_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """Returns:
        - eval/inference: logits tensor [B, C, H, W]
        - training with deep supervision: dict {'main': ..., 'aux': ...}
        """
        if self.deep_supervision and self.training:
            features = self.net.encoder(x)
            decoder_out = self.net.decoder(features)
            main_logits = self.net.segmentation_head(decoder_out)
            aux_logits = self.aux_head(features[-1])
            aux_logits = F.interpolate(
                aux_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            return {"main": main_logits, "aux": aux_logits}
        return self.net(x)


def deep_supervision_loss(loss_fn, outputs, targets, aux_weight: float = 0.4):
    """Compute compound loss across deep-supervision outputs.

    loss_fn signature: (logits, targets) -> (scalar, dict)
    outputs: dict from UNetPlusPlusModel during training, or raw tensor.
    """
    if isinstance(outputs, dict):
        main_loss, breakdown = loss_fn(outputs["main"], targets)
        aux_loss, _ = loss_fn(outputs["aux"], targets)
        total = main_loss + aux_weight * aux_loss
        breakdown["aux"] = float(aux_loss.detach().item())
        breakdown["total"] = float(total.detach().item())
        return total, breakdown
    return loss_fn(outputs, targets)
