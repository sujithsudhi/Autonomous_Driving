"""Train BEVFormer-Lite using a modular trainer."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.camera.bevformer_lite import BEVFormerLite
from train_datasets.nuscenes_bev import NuScenesBEVDataset, collate_bevformer
from training.config import TrainingConfig
from training.engine import BEVFormerLiteTrainer, TrainerState, write_history
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


def load_config(path: str) -> TrainingConfig:
    """Load a structured config matching the Transformers-style YAML layout."""

    return TrainingConfig.from_yaml(path)


def setup_wandb(cfg: TrainingConfig):
    """Initialize Weights & Biases run when requested."""

    if not cfg.logging.wandb_enabled or not is_main_process():
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipping experiment tracking.")
        return None

    run = wandb.init(
        project=cfg.logging.wandb_project or cfg.experiment.name,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.wandb_run_name or cfg.experiment.name,
        tags=cfg.logging.wandb_tags or None,
        config=cfg.to_dict(),
        resume="allow",
    )
    return run


def prepare_environment(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_dataloader(
    cfg: TrainingConfig, split: str, shuffle: bool, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = NuScenesBEVDataset(
        dataroot=cfg.dataset.root,
        version=cfg.dataset.version,
        cameras=cfg.dataset.cameras,
        bev_bounds=cfg.dataset.bev_bounds,
        split=split,
        image_size=cfg.dataset.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collate_bevformer,
    )


def build_dataloaders(cfg: TrainingConfig, distributed: bool) -> Tuple[DataLoader, DataLoader]:
    train_loader = build_dataloader(
        cfg,
        split=cfg.dataset.train_split,
        shuffle=not distributed,
        batch_size=cfg.optimization.batch_size,
        num_workers=cfg.optimization.num_workers,
    )
    val_loader = build_dataloader(
        cfg,
        split=cfg.dataset.val_split,
        shuffle=False,
        batch_size=cfg.optimization.val_batch_size or cfg.optimization.batch_size,
        num_workers=cfg.optimization.val_num_workers or cfg.optimization.num_workers,
    )
    return train_loader, val_loader


def build_model(cfg: TrainingConfig, device: torch.device, distributed: bool) -> torch.nn.Module:
    model = BEVFormerLite(
        bev_bounds=cfg.dataset.bev_bounds,
        bev_resolution=cfg.dataset.bev_resolution,
        num_cams=len(cfg.dataset.cameras),
        backbone_name=cfg.model.backbone,
        embed_dim=cfg.model.embed_dim,
        num_classes=cfg.model.num_classes,
        bev_encoder_layers=cfg.model.bev_encoder_layers,
        bev_num_heads=cfg.model.bev_num_heads,
        dropout=cfg.model.dropout,
        image_size=tuple(cfg.dataset.image_size),
        attn_chunk_size=cfg.model.cross_attn_chunk_size,
        max_attn_elements=cfg.model.max_cross_attn_elements,
        camera_token_stride=cfg.model.camera_token_stride,
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None
        )
    return model


def build_optimizer_and_scheduler(
    cfg: TrainingConfig, model: torch.nn.Module
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimization.lr,
        weight_decay=cfg.optimization.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.optimization.lr_reduce_factor,
        patience=cfg.optimization.lr_plateau_patience,
    )
    return optimizer, scheduler


def resume_if_available(trainer: BEVFormerLiteTrainer, path: str | None) -> TrainerState:
    if path:
        state = trainer.load_checkpoint(path)
        if is_main_process():
            print(
                f"Resumed from {path}: starting at epoch {state.epoch + 1}, "
                f"best val loss {state.best_val_loss:.4f}"
            )
        return state
    return TrainerState()


def log_epoch_summary(train_history, val_history) -> None:
    if not is_main_process() or not train_history:
        return
    print("\n=== Training & Validation Loss History ===")
    for idx, (trn, val) in enumerate(zip(train_history, val_history)):
        print(
            f"Epoch {idx+1:03d}: "
            f"train_total={trn['loss_total']:.4f}, train_cls={trn['loss_cls']:.4f}, "
            f"train_box={trn['loss_box']:.4f} | "
            f"val_total={val['loss_total']:.4f}, val_cls={val['loss_cls']:.4f}, "
            f"val_box={val['loss_box']:.4f}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    distributed = init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = setup_wandb(cfg)

    prepare_environment(device)

    train_loader, val_loader = build_dataloaders(cfg, distributed)
    model = build_model(cfg, device, distributed)
    criterion = BEVDetectionLoss(
        bev_bounds=cfg.dataset.bev_bounds,
        bev_resolution=cfg.dataset.bev_resolution,
        cls_weight=cfg.loss.cls_weight,
        box_weight=cfg.loss.box_weight,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

    use_amp = cfg.optimization.mixed_precision if device.type == "cuda" else False
    scaler = GradScaler(enabled=device.type == "cuda" and use_amp)

    trainer = BEVFormerLiteTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=Path(cfg.experiment.output_dir),
        early_stop_patience=cfg.optimization.early_stop_patience,
        max_grad_norm=cfg.optimization.max_grad_norm,
        use_amp=use_amp,
        distributed=distributed,
        wandb_run=wandb_run,
    )

    state = resume_if_available(trainer, args.resume)

    try:
        train_history, val_history, state = trainer.fit(
            num_epochs=cfg.optimization.epochs,
            state=state,
        )
    except Exception:
        if wandb_run is not None:
            # Ensure the run is closed even if training exits early.
            wandb_run.finish()
        raise

    barrier()
    log_epoch_summary(train_history, val_history)
    if is_main_process():
        output_dir = Path(cfg.experiment.output_dir)
        cfg.dump(output_dir / "config.yaml")
        write_history(output_dir, train_history, val_history, state)
        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = state.best_val_loss
            wandb_run.summary["best_epoch"] = state.best_epoch
            wandb_run.finish()


if __name__ == "__main__":
    main()
