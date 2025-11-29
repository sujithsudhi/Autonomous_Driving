"""Train BEVFormer-Lite using a modular trainer."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.camera.bevformer_lite import BEVFormerLite
from train_datasets.nuscenes_bev import NuScenesBEVDataset, collate_bevformer
from training.config import ResolutionStageConfig, TrainingConfig
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
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume from <output_dir>/best.pth instead of starting fresh.",
    )
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
        # Let cuDNN pick the fastest conv algorithms for the fixed input shapes
        torch.backends.cudnn.benchmark = True


def build_dataloader(
    cfg: TrainingConfig,
    split: str,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    image_size: Tuple[int, int] | None = None,
) -> DataLoader:
    dataset = NuScenesBEVDataset(
        dataroot=cfg.dataset.root,
        version=cfg.dataset.version,
        cameras=cfg.dataset.cameras,
        bev_bounds=cfg.dataset.bev_bounds,
        split=split,
        image_size=image_size or cfg.dataset.image_size,
    )
    sampler = (
        DistributedSampler(
            dataset, shuffle=shuffle, drop_last=shuffle
        )
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=shuffle,
        collate_fn=collate_bevformer,
    )


def build_dataloaders(
    cfg: TrainingConfig, distributed: bool, image_size: Tuple[int, int] | None = None
) -> Tuple[DataLoader, DataLoader]:
    train_loader = build_dataloader(
        cfg,
        split=cfg.dataset.train_split,
        shuffle=True,
        batch_size=cfg.optimization.batch_size,
        num_workers=cfg.optimization.num_workers,
        distributed=distributed,
        image_size=image_size,
    )
    val_loader = build_dataloader(
        cfg,
        split=cfg.dataset.val_split,
        shuffle=False,
        batch_size=cfg.optimization.val_batch_size or cfg.optimization.batch_size,
        num_workers=cfg.optimization.val_num_workers or cfg.optimization.num_workers,
        distributed=distributed,
        image_size=image_size,
    )
    return train_loader, val_loader


def resolve_schedule(cfg: TrainingConfig) -> List[ResolutionStageConfig]:
    if cfg.optimization.resolution_schedule:
        schedule: List[ResolutionStageConfig] = []
        for idx, stage in enumerate(cfg.optimization.resolution_schedule):
            if stage.warmup_epochs == 0 and cfg.optimization.warmup_epochs > 0 and idx == 0:
                stage.warmup_epochs = cfg.optimization.warmup_epochs
            if stage.freeze_backbone is None:
                stage.freeze_backbone = cfg.model.freeze_backbone
            schedule.append(stage)
        return schedule
    return [
        ResolutionStageConfig(
            epochs=cfg.optimization.epochs,
            image_size=list(cfg.dataset.image_size),
            freeze_backbone=cfg.model.freeze_backbone,
            warmup_epochs=cfg.optimization.warmup_epochs,
        )
    ]


def _stage_image_size(stage: ResolutionStageConfig, cfg: TrainingConfig) -> Tuple[int, int]:
    if stage.image_size is not None:
        return tuple(stage.image_size)
    return tuple(cfg.dataset.image_size)


def _set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if not hasattr(module, "backbone"):
        return
    for param in module.backbone.parameters():
        param.requires_grad = trainable


def _set_model_image_size(model: torch.nn.Module, image_size: Tuple[int, int]) -> None:
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if hasattr(module, "set_image_size"):
        module.set_image_size(image_size)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    optimizer.state.clear()


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
        head_dropout=cfg.model.head_dropout,
        image_size=tuple(cfg.dataset.image_size),
        attn_chunk_size=cfg.model.cross_attn_chunk_size,
        max_attn_elements=cfg.model.max_cross_attn_elements,
        camera_token_stride=cfg.model.camera_token_stride,
    ).to(device)
    # Calibrate objectness prior based on expected sparsity.
    model.head.init_obj_bias_from_prob(cfg.loss.obj_prior_prob)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None
        )
    return model


def build_optimizer(cfg: TrainingConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimization.lr,
        weight_decay=cfg.optimization.weight_decay,
    )
    return optimizer


def build_scheduler(
    cfg: TrainingConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.optimization.lr_reduce_factor,
        patience=cfg.optimization.lr_plateau_patience,
    )
    return scheduler


def resume_if_available(trainer: BEVFormerLiteTrainer, path: str | Path | None) -> TrainerState:
    if path:
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            if is_main_process():
                print(f"Resume path {ckpt_path} not found; starting from scratch.")
            return TrainerState()
        state = trainer.load_checkpoint(str(ckpt_path))
        if is_main_process():
            print(
                f"Resumed from {ckpt_path}: starting at epoch {state.epoch + 1}, "
                f"best val loss {state.best_val_loss:.4f}"
            )
        return state
    return TrainerState()


def log_epoch_summary(train_history, val_history) -> None:
    if not is_main_process() or not train_history:
        return
    print("\n=== Training & Validation Loss History ===")
    for idx, (trn, val) in enumerate(zip(train_history, val_history)):
        pos_str = ""
        if "num_positive" in val:
            pos_str = f", val_pos={val['num_positive']:.1f}"
        print(
            f"Epoch {idx+1:03d}: "
            f"train_total={trn['loss_total']:.4f}, train_obj={trn.get('loss_obj', float('nan')):.4f}, "
            f"train_cls={trn['loss_cls']:.4f}, train_box={trn['loss_box']:.4f} | "
            f"val_total={val['loss_total']:.4f}, val_obj={val.get('loss_obj', float('nan')):.4f}, "
            f"val_cls={val['loss_cls']:.4f}, val_box={val['loss_box']:.4f}{pos_str}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    distributed = init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = setup_wandb(cfg)

    prepare_environment(device)

    schedule = resolve_schedule(cfg)
    model = build_model(cfg, device, distributed)
    criterion = BEVDetectionLoss(
        bev_bounds=cfg.dataset.bev_bounds,
        bev_resolution=cfg.dataset.bev_resolution,
        cls_weight=cfg.loss.cls_weight,
        box_weight=cfg.loss.box_weight,
        obj_weight=cfg.loss.obj_weight,
        obj_pos_weight=cfg.loss.obj_pos_weight,
        neg_pos_ratio=cfg.loss.neg_pos_ratio,
        label_smoothing=cfg.loss.label_smoothing,
    )
    optimizer = build_optimizer(cfg, model)

    use_amp = cfg.optimization.mixed_precision if device.type == "cuda" else False
    scaler = GradScaler(enabled=device.type == "cuda" and use_amp)

    state = TrainerState()
    train_history_all: List[Dict[str, float]] = []
    val_history_all: List[Dict[str, float]] = []
    resumed = False
    completed_epochs = 0

    try:
        for stage_idx, stage in enumerate(schedule):
            stage_start = completed_epochs
            stage_end = stage_start + stage.epochs
            current_image_size = _stage_image_size(stage, cfg)

            if state.epoch >= stage_end:
                completed_epochs = stage_end
                continue

            _set_model_image_size(model, current_image_size)

            if stage.freeze_backbone is not None:
                _set_backbone_trainable(model, not stage.freeze_backbone)

            stage_base_lr = cfg.optimization.lr * stage.lr_scale
            starting_fresh = state.epoch == stage_start
            if starting_fresh:
                if stage.reset_optimizer:
                    _reset_optimizer_state(optimizer)
                _set_optimizer_lr(optimizer, stage_base_lr)
            scheduler = build_scheduler(cfg, optimizer)

            train_loader, val_loader = build_dataloaders(
                cfg, distributed, image_size=current_image_size
            )
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

            resume_path: str | None = args.resume
            if args.resume_best:
                resume_path = str(Path(cfg.experiment.output_dir) / "best.pth")
                if is_main_process():
                    print(f"Resume-best requested; will try loading {resume_path}")

            if not resumed:
                state = resume_if_available(trainer, resume_path)
                resumed = True

            if is_main_process():
                print(
                    f"\n--- Stage {stage_idx+1}/{len(schedule)}: epochs "
                    f"{state.epoch}->{stage_end}, image_size={current_image_size} ---"
                )

            train_history, val_history, state = trainer.fit(
                num_epochs=stage_end,
                state=state,
                stage_base_lr=stage_base_lr,
                stage_warmup_epochs=stage.warmup_epochs,
                stage_start_epoch=stage_start,
            )
            train_history_all.extend(train_history)
            val_history_all.extend(val_history)
            completed_epochs = stage_end
    except Exception:
        if wandb_run is not None:
            wandb_run.finish()
        raise

    barrier()
    log_epoch_summary(train_history_all, val_history_all)
    if is_main_process():
        output_dir = Path(cfg.experiment.output_dir)
        cfg.dump(output_dir / "config.yaml")
        write_history(output_dir, train_history_all, val_history_all, state)
        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = state.best_val_loss
            wandb_run.summary["best_epoch"] = state.best_epoch
            wandb_run.finish()


if __name__ == "__main__":
    main()
