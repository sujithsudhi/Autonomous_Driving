import argparse
import gc
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch import amp
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_datasets.nuscenes_bev import NuScenesBEVDataset, collate_bevformer
from models.camera.bevformer_lite import BEVFormerLite
from training.losses_bev import BEVDetectionLoss
from utils.distributed import barrier, init_distributed, is_main_process


def format_losses(losses: Dict[str, float]) -> str:
    return (
        f"total={losses.get('loss_total', float('nan')):.3f} "
        f"cls={losses.get('loss_cls', float('nan')):.3f} "
        f"box={losses.get('loss_box', float('nan')):.3f}"
    )


def display_progress(prefix: str, epoch: int, step: int, total: int, losses: Dict[str, float]) -> None:
    progress = f"[{prefix}] Epoch {epoch} {step}/{total} {format_losses(losses)}"
    print(progress, end="\r", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BEVFormer-Lite on nuScenes.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bevformer_lite_nuscenes.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    distributed = init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dataset = NuScenesBEVDataset(
        dataroot=cfg["dataset"]["root"],
        version=cfg["dataset"]["version"],
        cameras=cfg["dataset"]["cameras"],
        bev_bounds=cfg["dataset"]["bev_bounds"],
        split=cfg["dataset"].get("train_split", "train"),
        image_size=cfg["dataset"]["image_size"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["optimization"]["batch_size"],
        shuffle=not distributed,
        num_workers=cfg["optimization"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_bevformer,
    )
    val_split = cfg["dataset"].get("val_split", "val")
    val_dataset = NuScenesBEVDataset(
        dataroot=cfg["dataset"]["root"],
        version=cfg["dataset"]["version"],
        cameras=cfg["dataset"]["cameras"],
        bev_bounds=cfg["dataset"]["bev_bounds"],
        split=val_split,
        image_size=cfg["dataset"]["image_size"],
    )
    val_batch_size = cfg["optimization"].get("val_batch_size", cfg["optimization"]["batch_size"])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cfg["optimization"]["num_workers"],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_bevformer,
    )

    model = BEVFormerLite(
        bev_bounds=cfg["dataset"]["bev_bounds"],
        bev_resolution=cfg["dataset"]["bev_resolution"],
        num_cams=len(cfg["dataset"]["cameras"]),
        backbone_name=cfg["model"]["backbone"],
        embed_dim=cfg["model"]["embed_dim"],
        num_classes=cfg["model"]["num_classes"],
        bev_encoder_layers=cfg["model"]["bev_encoder_layers"],
        bev_num_heads=cfg["model"]["bev_num_heads"],
        dropout=cfg["model"]["dropout"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        attn_chunk_size=cfg["model"].get("cross_attn_chunk_size", 256),
        max_attn_elements=cfg["model"].get("max_cross_attn_elements", 25_000_000),
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None
        )

    criterion = BEVDetectionLoss(
        bev_bounds=cfg["dataset"]["bev_bounds"],
        bev_resolution=cfg["dataset"]["bev_resolution"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimization"]["lr"],
        weight_decay=cfg["optimization"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["optimization"]["epochs"])
    use_amp = cfg["optimization"].get("mixed_precision", device.type == "cuda")
    scaler = GradScaler(enabled=device.type == "cuda" and use_amp)

    def autocast_context():
        if device.type == "cuda":
            return amp.autocast(device_type="cuda", enabled=scaler.is_enabled())
        return nullcontext()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_history = []
    val_history = []

    for epoch in range(start_epoch, cfg["optimization"]["epochs"]):
        model.train()
        epoch_totals = {"loss_total": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        num_train_batches = 0
        for step, batch in enumerate(dataloader):
            images = batch["images"].to(device)
            gt_boxes = batch["gt_boxes"].to(device)
            gt_labels = batch["gt_labels"].to(device)
            gt_masks = batch["gt_masks"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                preds = model(images)
                losses = criterion(
                    preds["cls_logits"],
                    preds["box_preds"],
                    gt_boxes,
                    gt_labels,
                    gt_masks,
                )
                loss_total = losses["loss_total"]

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            num_train_batches += 1
            for key in epoch_totals.keys():
                epoch_totals[key] += losses[key].item()

            if is_main_process():
                current_avg = {
                    key: epoch_totals[key] / max(num_train_batches, 1) for key in epoch_totals.keys()
                }
                display_progress("Train", epoch + 1, step + 1, len(dataloader), current_avg)

        train_epoch = {
            key: (epoch_totals[key] / max(num_train_batches, 1)) for key in epoch_totals.keys()
        }
        train_history.append(train_epoch)

        if is_main_process():
            print()

        model.eval()
        val_totals = {"loss_total": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        val_counts = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_dataloader):
                images = batch["images"].to(device)
                gt_boxes = batch["gt_boxes"].to(device)
                gt_labels = batch["gt_labels"].to(device)
                gt_masks = batch["gt_masks"].to(device)

                with autocast_context():
                    preds = model(images)
                    losses = criterion(
                        preds["cls_logits"],
                        preds["box_preds"],
                        gt_boxes,
                        gt_labels,
                        gt_masks,
                    )
                val_counts += 1
                for key in val_totals.keys():
                    val_totals[key] += losses[key].item()

                if is_main_process():
                    current_avg = {
                        key: val_totals[key] / max(val_counts, 1) for key in val_totals.keys()
                    }
                    display_progress("Val", epoch + 1, val_step + 1, len(val_dataloader), current_avg)

        if is_main_process():
            print()

        val_epoch = {}
        if val_counts > 0:
            val_epoch = {key: val_totals[key] / val_counts for key in epoch_totals.keys()}
        else:
            val_epoch = {key: float("nan") for key in epoch_totals.keys()}
        val_history.append(val_epoch)

        if is_main_process():
            print(
                f"[Epoch {epoch+1}] Train total={train_epoch['loss_total']:.4f}, "
                f"Val total={val_epoch['loss_total']:.4f}"
            )

        scheduler.step()
        if is_main_process() and (epoch + 1) % cfg["logging"]["checkpoint_interval"] == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.module.state_dict() if distributed else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            ckpt_path = output_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        barrier()

    if is_main_process() and train_history:
        print("\n=== Training & Validation Loss History ===")
        for idx, (trn, val) in enumerate(zip(train_history, val_history)):
            print(
                f"Epoch {idx+1:03d}: "
                f"train_total={trn['loss_total']:.4f}, train_cls={trn['loss_cls']:.4f}, "
                f"train_box={trn['loss_box']:.4f} | "
                f"val_total={val['loss_total']:.4f}, val_cls={val['loss_cls']:.4f}, "
                f"val_box={val['loss_box']:.4f}"
            )


if __name__ == "__main__":
    main()
