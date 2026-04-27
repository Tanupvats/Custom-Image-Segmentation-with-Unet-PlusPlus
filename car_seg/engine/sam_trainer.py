
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from car_seg.data.sam_dataset import build_sam_loader
from car_seg.engine.optim import build_optimizer, build_scheduler
from car_seg.losses import build_loss
from car_seg.models.sam_model import SAMModel
from car_seg.utils.checkpoint import save_checkpoint, load_checkpoint
from car_seg.utils.logging import MetricLogger, setup_stdout_logger
from car_seg.utils.metrics import binary_iou


def train_sam(cfg, resume: str | None = None) -> None:
    log = setup_stdout_logger("car_seg.sam_trainer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.experiment.seed)

    out_dir = Path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = MetricLogger(
        output_dir=out_dir,
        use_tensorboard=cfg.logging.tensorboard,
        use_wandb=cfg.logging.wandb,
        wandb_project=cfg.logging.wandb_project,
        run_name=cfg.experiment.name,
        config=cfg.to_dict(),
    )

    train_loader = build_sam_loader(cfg, train=True)
    val_loader = build_sam_loader(cfg, train=False)
    log.info(f"Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = SAMModel(cfg).to(device)
    n_params_all = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SAM total params: {n_params_all/1e6:.1f}M, trainable: {n_params_trainable/1e6:.2f}M "
             f"({100*n_params_trainable/max(n_params_all,1):.2f}%)")

    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    accum_steps = max(int(cfg.train.get("accum_steps", 1)), 1)
    scheduler = build_scheduler(
        optimizer, cfg,
        steps_per_epoch=max(len(train_loader) // accum_steps, 1),
    )
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_metric = -math.inf
    if resume is not None:
        ckpt = load_checkpoint(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", -math.inf)

    global_step = 0
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        t0 = time.time()
        running = {"total": 0.0}
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            gt_masks = batch["binary_mask"].to(device, non_blocking=True)  # [B,H,W]
            prompts = [{k: v.to(device) for k, v in p.items()} for p in batch["prompt"]]

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(images, prompts)
                logits = out["logits"]  # [B, M, H, W]; M=1 with our cfg
                # Squeeze the multimask dim → [B, H, W]
                logits_b = logits[:, 0]
                loss, breakdown = loss_fn(logits_b, gt_masks)
                loss = loss / accum_steps

            scaler.scale(loss).backward() if loss.requires_grad else None

            if (step + 1) % accum_steps == 0:
                if cfg.train.get("grad_clip_norm") is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.train.grad_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                global_step += 1

            bs = images.size(0)
            n_seen += bs
            for k, v in breakdown.items():
                running[k] = running.get(k, 0.0) + v * bs * accum_steps  # undo division

            if (step + 1) % cfg.train.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_total = running["total"] / max(n_seen, 1)
                pbar.set_postfix(loss=f"{avg_total:.4f}", lr=f"{lr:.2e}")
                metric_logger.log({"train/total": avg_total, "train/lr": lr}, step=global_step)

        avg = {k: v / max(n_seen, 1) for k, v in running.items()}
        log.info(f"Epoch {epoch+1} train: loss={avg['total']:.4f} time={time.time()-t0:.1f}s")

        # ---- Validation: per-class IoU using center-of-mass point prompts ----
        val_iou = evaluate_sam(model, val_loader, device, use_amp=use_amp)
        log.info(f"Epoch {epoch+1} val mIoU (binary, prompted): {val_iou:.4f}")
        metric_logger.log({"val/miou": val_iou, "epoch": epoch + 1}, step=global_step)

        # ---- Checkpointing ----
        save_checkpoint(
            out_dir / "last.pt",
            model_sd=model.state_dict(),
            optimizer_sd=optimizer.state_dict(),
            scaler_sd=scaler.state_dict() if use_amp else None,
            epoch=epoch,
            best_metric=best_metric,
            extra={"config": cfg.to_dict()},
        )
        if val_iou > best_metric:
            best_metric = val_iou
            save_checkpoint(
                out_dir / "best.pt",
                model_sd=model.state_dict(),
                epoch=epoch,
                best_metric=best_metric,
                extra={"config": cfg.to_dict()},
            )
            log.info(f"  ↳ new best mIoU={best_metric:.4f}, saved best.pt")

    metric_logger.close()
    log.info(f"SAM training done. Best mIoU={best_metric:.4f}")


@torch.no_grad()
def evaluate_sam(model, val_loader, device, use_amp: bool = True) -> float:
    """SAM eval: take each (image, prompt, gt) and report mean binary IoU.

    The dataset has already sampled one prompt per image; here we just feed it
    through and measure IoU. For more thorough eval you'd run multiple prompts
    per class, but this is a fair training signal.
    """
    model.eval()
    ious: list[float] = []
    for batch in tqdm(val_loader, desc="val (sam)", dynamic_ncols=True, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        gt = batch["binary_mask"].to(device, non_blocking=True)
        prompts = [{k: v.to(device) for k, v in p.items()} for p in batch["prompt"]]
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            out = model(images, prompts)
        pred = (out["logits"][:, 0] > 0).float()  # logit > 0 ⇔ sigmoid > 0.5
        for b in range(pred.size(0)):
            iou = binary_iou(pred[b], gt[b])
            if not np.isnan(iou):
                ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0
