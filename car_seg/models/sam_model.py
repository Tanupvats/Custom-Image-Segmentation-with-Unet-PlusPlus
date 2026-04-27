
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry


class SAMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ckpt = cfg.model.sam_checkpoint
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {ckpt}. Download from "
                "https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        self.sam = sam_model_registry[cfg.model.sam_model_type](checkpoint=ckpt)
        self.image_size = cfg.data.image_size  # 1024 for SAM

        if cfg.model.freeze_image_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad_(False)
            self.sam.image_encoder.eval()

        if cfg.model.freeze_prompt_encoder:
            for p in self.sam.prompt_encoder.parameters():
                p.requires_grad_(False)
            self.sam.prompt_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen encoder in eval() so BN/dropout don't drift
        if self.cfg.model.freeze_image_encoder:
            self.sam.image_encoder.eval()
        if self.cfg.model.freeze_prompt_encoder:
            self.sam.prompt_encoder.eval()
        return self

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Run the (typically frozen) image encoder. images: [B,3,1024,1024]."""
        return self.sam.image_encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        prompts: Sequence[dict],
        multimask_output: bool | None = None,
    ) -> dict:
        """One forward step on a batch of (image, prompts) pairs.

        Args:
            images: [B,3,H,W] tensor pre-normalized by SAM's expected stats
                (the dataset handles this). H=W=1024.
            prompts: list of length B; each dict has any subset of:
                point_coords [N,2], point_labels [N],
                boxes [K,4], mask_inputs [1,1,256,256].
            multimask_output: if None, uses cfg default.

        Returns:
            dict with:
                "logits": [B, M, H_out, W_out] mask logits at full resolution
                          (M=1 if not multimask, else 3)
                "iou_predictions": [B, M] predicted mask quality
        """
        if multimask_output is None:
            multimask_output = self.cfg.model.multimask_output

        # If image encoder is frozen we can use no_grad for that branch
        if self.cfg.model.freeze_image_encoder:
            with torch.no_grad():
                image_embeddings = self.sam.image_encoder(images)
        else:
            image_embeddings = self.sam.image_encoder(images)

        all_logits = []
        all_iou = []
        for b in range(images.size(0)):
            p = prompts[b]
            point_coords = p.get("point_coords")
            point_labels = p.get("point_labels")
            boxes = p.get("boxes")
            mask_inputs = p.get("mask_inputs")

            points = None
            if point_coords is not None and point_labels is not None:
                # SAM expects shapes [B_p, N, 2] / [B_p, N] with B_p=1 here
                pc = point_coords.unsqueeze(0) if point_coords.dim() == 2 else point_coords
                pl = point_labels.unsqueeze(0) if point_labels.dim() == 1 else point_labels
                points = (pc, pl)

            if boxes is not None and boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)

            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=points, boxes=boxes, masks=mask_inputs
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings[b : b + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=multimask_output,
            )

            # Upsample to input resolution
            high_res = F.interpolate(
                low_res_masks, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
            all_logits.append(high_res)
            all_iou.append(iou_predictions)

        return {
            "logits": torch.cat(all_logits, dim=0),       # [B, M, H, W]
            "iou_predictions": torch.cat(all_iou, dim=0), # [B, M]
        }
