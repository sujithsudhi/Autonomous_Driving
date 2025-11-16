import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_datasets.nuscenes_bev import NuScenesBEVDataset, collate_bevformer
from models.camera.bevformer_lite import BEVFormerLite
from training.losses_bev import BEVDetectionLoss
from utils.distributed import barrier, init_distributed, is_main_process


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


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: BEVDetectionLoss,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    loss_components = {"loss_cls": 0.0, "loss_box": 0.0}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            gt_boxes = batch["gt_boxes"].to(device)
            gt_labels = batch["gt_labels"].to(device)
            gt_masks = batch["gt_masks"].to(device)

            batch_size = images.size(0)
            with autocast(enabled=use_amp):
                preds = model(images)
                losses = criterion(
                    preds["cls_logits"],
                    preds["box_preds"],
                    gt_boxes,
                    gt_labels,
                    gt_masks,
                )

            total_loss += losses["loss_total"].item() * batch_size
            loss_components["loss_cls"] += losses["loss_cls"].item() * batch_size
            loss_components["loss_box"] += losses["loss_box"].item() * batch_size
            total_items += batch_size

    if dist.is_initialized():
        buffer = torch.tensor(
            [total_loss, loss_components["loss_cls"], loss_components["loss_box"], total_items],
            device=device,
        )
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        total_loss, cls_sum, box_sum, total_items = buffer.tolist()
        loss_components = {"loss_cls": cls_sum, "loss_box": box_sum}

    denom = max(total_items, 1)
    averaged = {k: v / denom for k, v in loss_components.items()}
    return total_loss / denom, averaged


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
        split="train",
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

    val_dataset = NuScenesBEVDataset(
        dataroot=cfg["dataset"]["root"],
        version=cfg["dataset"]["version"],
        cameras=cfg["dataset"]["cameras"],
        bev_bounds=cfg["dataset"]["bev_bounds"],
        split="val",
        image_size=cfg["dataset"]["image_size"],
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["optimization"].get("val_batch_size", cfg["optimization"]["batch_size"]),
        shuffle=False,
        num_workers=cfg["optimization"].get("val_num_workers", cfg["optimization"]["num_workers"]),
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
        camera_token_stride=cfg["model"].get("camera_token_stride", 1),
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
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["optimization"].get("lr_reduce_factor", 0.5),
        patience=cfg["optimization"].get("lr_plateau_patience", 5),
        verbose=is_main_process(),
    )
    use_amp = cfg["optimization"].get("mixed_precision", device.type == "cuda")
    scaler = GradScaler(enabled=device.type == "cuda" and use_amp)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epochs_since_improvement = checkpoint.get("epochs_since_improvement", 0)
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        epochs_since_improvement = 0

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    early_stop_patience = cfg["optimization"].get("early_stop_patience", 10)

    for epoch in range(start_epoch, cfg["optimization"]["epochs"]):
        model.train()
        running_losses = {"loss_total": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        for step, batch in enumerate(dataloader):
            images = batch["images"].to(device)
            gt_boxes = batch["gt_boxes"].to(device)
            gt_labels = batch["gt_labels"].to(device)
            gt_masks = batch["gt_masks"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
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

            for key in running_losses.keys():
                running_losses[key] += losses[key].item()
            if is_main_process() and (step + 1) % cfg["logging"]["print_interval"] == 0:
                denom = cfg["logging"]["print_interval"]
                print(
                    f"Epoch {epoch+1} Step {step+1}: "
                    f"total={running_losses['loss_total']/denom:.3f} "
                    f"cls={running_losses['loss_cls']/denom:.3f} "
                    f"box={running_losses['loss_box']/denom:.3f}"
                )
                for key in running_losses.keys():
                    running_losses[key] = 0.0

        val_loss, val_components = validate(model, val_dataloader, criterion, device, use_amp)
        scheduler.step(val_loss)

        if is_main_process():
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1} validation: loss={val_loss:.3f} "
                f"cls={val_components['loss_cls']:.3f} "
                f"box={val_components['loss_box']:.3f} lr={lr:.6f}"
            )

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            if is_main_process():
                checkpoint = {
                    "epoch": epoch + 1,
                    "model": model.module.state_dict() if distributed else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epochs_since_improvement": epochs_since_improvement,
                }
                ckpt_path = output_dir / "best.pth"
                torch.save(checkpoint, ckpt_path)
                print(f"Saved best checkpoint to {ckpt_path}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stop_patience:
                if is_main_process():
                    print(
                        f"Early stopping at epoch {epoch+1} after "
                        f"{epochs_since_improvement} epochs without improvement."
                    )
                break

        barrier()


if __name__ == "__main__":
    main()
