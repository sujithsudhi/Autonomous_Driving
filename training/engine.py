"""Training utilities for BEVFormer-Lite."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.distributed import barrier, is_main_process

if TYPE_CHECKING:
    import wandb


@dataclass
class TrainerState:
    """Container for training loop state and early stopping."""

    epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_since_improvement: int = 0
    best_epoch: Optional[int] = None


def format_losses(losses: Dict[str, float]) -> str:
    parts = [
        f"total={losses.get('loss_total', float('nan')):.3f}",
        f"obj={losses.get('loss_obj', float('nan')):.3f}",
        f"cls={losses.get('loss_cls', float('nan')):.3f}",
        f"box={losses.get('loss_box', float('nan')):.3f}",
    ]
    return " ".join(parts)


def write_history(
    output_dir: Path,
    train_history: Iterable[Dict[str, float]],
    val_history: Iterable[Dict[str, float]],
    state: TrainerState,
) -> None:
    """Persist loss curves to disk for quick experiment review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "epoch,train_total,train_obj,train_cls,train_box,"
            "val_total,val_obj,val_cls,val_box\n"
        )
        for idx, (trn, val) in enumerate(zip(train_history, val_history)):
            handle.write(
                f"{idx+1},"
                f"{trn['loss_total']:.6f},{trn.get('loss_obj', float('nan')):.6f},"
                f"{trn['loss_cls']:.6f},{trn['loss_box']:.6f},"
                f"{val['loss_total']:.6f},{val.get('loss_obj', float('nan')):.6f},"
                f"{val['loss_cls']:.6f},{val['loss_box']:.6f}\n"
            )

        best_epoch_str = str(state.best_epoch) if state.best_epoch is not None else "N/A"
        best_val_loss_str = (
            f"{state.best_val_loss:.6f}" if state.best_val_loss < float("inf") else "N/A"
        )
        handle.write(
            f"\nBest epoch: {best_epoch_str}\nBest val loss: {best_val_loss_str}\n"
        )

    print(f"Saved training metrics to {metrics_path}")


class BEVFormerLiteTrainer:
    """Encapsulates train/eval loops and checkpoint management."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        scaler: GradScaler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        early_stop_patience: int,
        max_grad_norm: float,
        use_amp: bool,
        distributed: bool,
        wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.early_stop_patience = early_stop_patience
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.distributed = distributed
        self.wandb_run = wandb_run

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _autocast_context(self):
        if self.device.type == "cuda":
            return autocast(device_type="cuda", enabled=self.scaler.is_enabled())
        return nullcontext()

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device_keys = {"images", "gt_boxes", "gt_labels", "gt_masks"}
        moved: Dict[str, torch.Tensor] = {}
        for key, tensor in batch.items():
            if key in device_keys:
                moved[key] = tensor.to(self.device, non_blocking=True)
            else:
                moved[key] = tensor
        return moved

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with self._autocast_context():
            preds = self.model(batch["images"])
            losses = self.criterion(
                preds["cls_logits"],
                preds["obj_logits"],
                preds["box_preds"],
                batch["gt_boxes"],
                batch["gt_labels"],
                batch["gt_masks"],
            )
        return losses

    def load_checkpoint(self, path: str) -> TrainerState:
        checkpoint = torch.load(path, map_location="cpu")
        state = TrainerState(
            epoch=checkpoint.get("epoch", 0),
            best_val_loss=checkpoint.get("best_val_loss", float("inf")),
            epochs_since_improvement=checkpoint.get("epochs_since_improvement", 0),
            best_epoch=checkpoint.get("best_epoch"),
        )

        if self.distributed:
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        return state

    def _save_checkpoint(self, state: TrainerState) -> None:
        if not is_main_process():
            return
        checkpoint = {
            "epoch": state.epoch,
            "model": self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": state.best_val_loss,
            "epochs_since_improvement": state.epochs_since_improvement,
            "best_epoch": state.best_epoch,
        }
        ckpt_path = self.output_dir / "best.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"Saved best checkpoint to {ckpt_path}")

    def _aggregate_losses(self, totals: Dict[str, float], counts: int) -> Dict[str, float]:
        return {key: totals[key] / max(counts, 1) for key in totals.keys()}

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _log_epoch_metrics(
        self, epoch: int, train_epoch: Dict[str, float], val_epoch: Dict[str, float], lr: float
    ) -> None:
        if self.wandb_run is None or not is_main_process():
            return

        log_payload = {
            "epoch": epoch,
            "train/loss_total": train_epoch.get("loss_total"),
            "train/loss_obj": train_epoch.get("loss_obj"),
            "train/loss_cls": train_epoch.get("loss_cls"),
            "train/loss_box": train_epoch.get("loss_box"),
            "val/loss_total": val_epoch.get("loss_total"),
            "val/loss_obj": val_epoch.get("loss_obj"),
            "val/loss_cls": val_epoch.get("loss_cls"),
            "val/loss_box": val_epoch.get("loss_box"),
            "train/num_positive": train_epoch.get("num_positive"),
            "val/num_positive": val_epoch.get("num_positive"),
            "lr": lr,
        }
        self.wandb_run.log(log_payload, step=epoch)

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_totals = {"loss_total": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        num_batches = 0
        pos_total = 0.0

        train_progress = tqdm(
            self.train_loader,
            desc="Train",
            total=len(self.train_loader),
            dynamic_ncols=True,
            leave=False,
            disable=not is_main_process(),
        )

        for batch in train_progress:
            batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            losses = self._forward(batch)
            loss_total = losses["loss_total"]

            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            num_batches += 1
            pos_total += float(losses.get("num_positive", 0.0))
            for key in epoch_totals.keys():
                epoch_totals[key] += losses[key].item()

            if is_main_process():
                current_avg = self._aggregate_losses(epoch_totals, num_batches)
                pos_avg = pos_total / max(num_batches, 1)
                train_progress.set_postfix_str(f"{format_losses(current_avg)} pos={pos_avg:.1f}")

        if is_main_process():
            train_progress.close()

        averaged = self._aggregate_losses(epoch_totals, num_batches)
        averaged["num_positive"] = pos_total / max(num_batches, 1)
        return averaged

    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        loss_totals = {"loss_total": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        pos_total = 0.0
        num_batches = 0

        val_progress = tqdm(
            self.val_loader,
            desc="Val",
            total=len(self.val_loader),
            dynamic_ncols=True,
            leave=False,
            disable=not is_main_process(),
        )

        with torch.no_grad():
            for batch in val_progress:
                batch = self._move_batch_to_device(batch)
                with self._autocast_context():
                    preds = self.model(batch["images"])
                    losses = self.criterion(
                        preds["cls_logits"],
                        preds["obj_logits"],
                        preds["box_preds"],
                        batch["gt_boxes"],
                        batch["gt_labels"],
                        batch["gt_masks"],
                    )

                pos_total += float(losses.get("num_positive", 0.0))
                num_batches += 1
                for key in loss_totals.keys():
                    loss_totals[key] += losses[key].item()

                if is_main_process():
                    averaged = self._aggregate_losses(loss_totals, num_batches)
                    pos_avg = pos_total / max(num_batches, 1)
                    val_progress.set_postfix_str(
                        f"{format_losses(averaged)} pos={pos_avg:.1f}"
                    )

        if is_main_process():
            val_progress.close()

        if dist.is_initialized():
            buffer = torch.tensor(
                [
                    loss_totals["loss_total"],
                    loss_totals["loss_obj"],
                    loss_totals["loss_cls"],
                    loss_totals["loss_box"],
                    pos_total,
                    num_batches,
                ],
                device=self.device,
            )
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            (
                total_loss,
                obj_sum,
                cls_sum,
                box_sum,
                pos_total,
                num_batches,
            ) = buffer.tolist()
            loss_totals = {
                "loss_total": total_loss,
                "loss_obj": obj_sum,
                "loss_cls": cls_sum,
                "loss_box": box_sum,
            }

        denom = max(num_batches, 1)
        averaged = {k: v / denom for k, v in loss_totals.items()}
        averaged["num_positive"] = pos_total / max(num_batches, 1)
        return averaged["loss_total"], averaged

    def fit(
        self,
        num_epochs: int,
        state: Optional[TrainerState] = None,
        stage_base_lr: float = 0.0,
        stage_warmup_epochs: int = 0,
        stage_start_epoch: int = 0,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], TrainerState]:
        if state is None:
            state = TrainerState()

        train_history: List[Dict[str, float]] = []
        val_history: List[Dict[str, float]] = []

        for epoch in range(state.epoch, num_epochs):
            stage_epoch = epoch - stage_start_epoch
            if stage_warmup_epochs > 0 and stage_epoch < stage_warmup_epochs:
                warmup_factor = (stage_epoch + 1) / max(stage_warmup_epochs, 1)
                self._set_lr(stage_base_lr * warmup_factor)
            elif stage_warmup_epochs > 0 and stage_epoch == stage_warmup_epochs:
                self._set_lr(stage_base_lr)

            train_sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_epoch = self.train_one_epoch()
            train_history.append(train_epoch)

            val_loss, val_components = self.evaluate()
            val_epoch = {"loss_total": val_loss, **val_components}
            val_history.append(val_epoch)
            self.scheduler.step(val_loss)

            state.epoch = epoch + 1

            lr = self.optimizer.param_groups[0]["lr"]
            if is_main_process():
                print(
                    f"Epoch {epoch+1} validation: loss={val_loss:.3f} "
                    f"obj={val_components['loss_obj']:.3f} "
                    f"cls={val_components['loss_cls']:.3f} "
                    f"box={val_components['loss_box']:.3f} "
                    f"pos={val_components.get('num_positive', float('nan')):.1f} lr={lr:.6f}"
                )

            if val_loss + 1e-6 < state.best_val_loss:
                state.best_val_loss = val_loss
                state.epochs_since_improvement = 0
                state.best_epoch = epoch + 1
                self._save_checkpoint(state)
            else:
                state.epochs_since_improvement += 1
                if state.epochs_since_improvement >= self.early_stop_patience:
                    if is_main_process():
                        print(
                            f"Early stopping at epoch {epoch+1} after "
                            f"{state.epochs_since_improvement} epochs without improvement."
                        )
                    break

            self._log_epoch_metrics(epoch + 1, train_epoch, val_epoch, lr)

            barrier()

        return train_history, val_history, state
