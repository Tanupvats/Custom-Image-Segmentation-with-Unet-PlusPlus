
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from car_seg.data import build_train_loader, build_val_loader
from car_seg.data.transforms import build_train_transform, build_val_transform
from car_seg.engine.optim import build_optimizer, build_scheduler
from car_seg.losses import build_loss
from car_seg.models import build_model
from car_seg.models.unetpp import deep_supervision_loss
from car_seg.utils.checkpoint import save_checkpoint, load_checkpoint
from car_seg.utils.ema import ModelEMA
from car_seg.utils.logging import MetricLogger, setup_stdout_logger
from car_seg.utils.metrics import SegmentationMetrics


def train_semantic(cfg, resume: str | None = None) -> None:
    log = setup_stdout_logger("car_seg.trainer")
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

    train_tf = build_train_transform(cfg)
    val_tf = build_val_transform(cfg)
    train_loader = build_train_loader(cfg, train_tf)
    val_loader = build_val_loader(cfg, val_tf)
    log.info(f"Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params: {n_params/1e6:.2f}M")

    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema = None
    if cfg.train.get("ema_decay") not in (None, 0, 0.0):
        ema = ModelEMA(model, decay=cfg.train.ema_decay, device=device)

    start_epoch = 0
    best_metric = -math.inf
    if resume is not None:
        ckpt = load_checkpoint(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except Exception: pass
        if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and ema is not None: ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", -math.inf)
        log.info(f"Resumed from {resume} at epoch {start_epoch}, best mIoU={best_metric:.4f}")

    grad_clip = cfg.train.get("grad_clip_norm")
    aux_weight = 0.4  # for UNet++ deep supervision
    is_unetpp_with_ds = (cfg.model.family == "unetpp" and cfg.model.get("deep_supervision", False))
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        t0 = time.time()
        running = {"total": 0.0}
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                if is_unetpp_with_ds:
                    loss, breakdown = deep_supervision_loss(loss_fn, outputs, masks, aux_weight=aux_weight)
                else:
                    loss, breakdown = loss_fn(outputs, masks)

            scaler.scale(loss).backward() if loss.requires_grad else None
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            if ema is not None:
                ema.update(model)

            bs = images.size(0)
            n_seen += bs
            for k, v in breakdown.items():
                running[k] = running.get(k, 0.0) + v * bs

            if (step + 1) % cfg.train.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg = {k: v / n_seen for k, v in running.items()}
                pbar.set_postfix(loss=f"{avg['total']:.4f}", lr=f"{lr:.2e}")
                metric_logger.log({**{f"train/{k}": v for k, v in avg.items()}, "train/lr": lr},
                                  step=global_step)
            global_step += 1

        train_avg = {k: v / max(n_seen, 1) for k, v in running.items()}
        epoch_time = time.time() - t0
        log.info(f"Epoch {epoch+1} train: loss={train_avg['total']:.4f} time={epoch_time:.1f}s")

        # ---- Validation ----
        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate_semantic(eval_model, val_loader, device, cfg, use_amp=use_amp)
        miou = val_metrics.miou
        log.info(
            f"Epoch {epoch+1} val: mIoU={miou:.4f} fwIoU={val_metrics.fwiou:.4f} "
            f"pixAcc={val_metrics.pixel_acc:.4f}"
        )
        log.info("\n" + val_metrics.format_table(top_k=10))

        # Step plateau scheduler with the val metric
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(miou)

        metric_logger.log(
            {"val/miou": miou, "val/fwiou": val_metrics.fwiou,
             "val/pixel_acc": val_metrics.pixel_acc, "val/mean_dice": val_metrics.mean_dice,
             "epoch": epoch + 1},
            step=global_step,
        )

        # ---- Checkpointing ----
        save_checkpoint(
            out_dir / "last.pt",
            model_sd=model.state_dict(),
            optimizer_sd=optimizer.state_dict(),
            scheduler_sd=getattr(scheduler, "state_dict", lambda: None)() if not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None,
            scaler_sd=scaler.state_dict() if use_amp else None,
            ema_sd=ema.state_dict() if ema is not None else None,
            epoch=epoch,
            best_metric=best_metric,
            extra={"config": cfg.to_dict()},
        )
        if miou > best_metric:
            best_metric = miou
            save_checkpoint(
                out_dir / "best.pt",
                model_sd=(ema.state_dict() if ema is not None else model.state_dict()),
                epoch=epoch,
                best_metric=best_metric,
                extra={"config": cfg.to_dict(), "metrics": val_metrics.to_summary()},
            )
            log.info(f"  ↳ new best mIoU={best_metric:.4f}, saved best.pt")

    metric_logger.close()
    log.info(f"Training done. Best mIoU={best_metric:.4f}")


@torch.no_grad()
def evaluate_semantic(model, val_loader: DataLoader, device, cfg, use_amp: bool = True):
    """Compute streaming mIoU + per-class IoU on the val loader."""
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=cfg.model.num_classes,
        ignore_index=cfg.data.ignore_index,
    )
    for batch in tqdm(val_loader, desc="val", dynamic_ncols=True, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["main"]
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
    return metrics.compute()
