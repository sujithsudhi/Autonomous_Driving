"""Visualize nuScenes samples with GT boxes and optional model predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple
import sys

import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.camera.bevformer_lite import BEVFormerLite
from train_datasets.nuscenes_bev import NuScenesBEVDataset
from training.config import TrainingConfig
from utils.geometry import project_points

matplotlib.use("Agg")


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize nuScenes BEV samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bevformer_lite_nuscenes.yaml",
        help="Training config to reuse dataset/model settings.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "mini"], help="Dataset split.")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Sample indices to visualize.")
    parser.add_argument("--num-samples", type=int, default=2, help="If indices not provided, visualize the first N.")
    parser.add_argument("--output-dir", type=str, default="viz/renders", help="Directory to save visualizations.")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Optional checkpoint to plot predictions.")
    parser.add_argument("--obj-threshold", type=float, default=0.35, help="Objectness threshold for predictions.")
    parser.add_argument("--topk", type=int, default=40, help="Max predictions to draw per sample.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    return parser.parse_args()


def build_dataset(cfg: TrainingConfig, split: str) -> NuScenesBEVDataset:
    return NuScenesBEVDataset(
        dataroot=cfg.dataset.root,
        version=cfg.dataset.version,
        cameras=cfg.dataset.cameras,
        bev_bounds=cfg.dataset.bev_bounds,
        split=split,
        image_size=cfg.dataset.image_size,
    )


def load_model(cfg: TrainingConfig, checkpoint: str, device: torch.device) -> BEVFormerLite:
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
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def unnormalize(image: torch.Tensor) -> np.ndarray:
    img = image * STD.to(image.device) + MEAN.to(image.device)
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def box_corners(box: torch.Tensor) -> torch.Tensor:
    x, y, z, l, w, h, yaw = box.tolist()
    dx, dy, dz = l / 2, w / 2, h / 2
    corners = torch.tensor(
        [
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
        ],
        dtype=torch.float32,
        device=box.device,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32, device=box.device)
    rotated = corners @ rot.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    rotated[:, 2] += z
    return rotated


def project_box_to_image(
    box: torch.Tensor, intr: torch.Tensor, extr: torch.Tensor, image_size: Tuple[int, int]
) -> np.ndarray | None:
    corners = box_corners(box)
    # depth filtering: require all corners in front of the camera
    cam_space = (extr @ torch.cat([corners, torch.ones_like(corners[:, :1])], dim=-1).T).T[:, 2]
    if (cam_space > 0.1).sum() < 8:
        return None
    pix = project_points(corners, intr, extr)
    w, h = image_size
    inside = (
        (pix[:, 0] >= 0)
        & (pix[:, 0] <= w - 1)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] <= h - 1)
    )
    if not inside.all():
        return None
    return pix.cpu().numpy()


def draw_box(ax, pix: np.ndarray, color: str, label: str | None = None) -> None:
    edges = [
        [0, 1, 2, 3, 0],  # bottom
        [4, 5, 6, 7, 4],  # top
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    for edge in edges:
        pts = pix[edge]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)
    if label:
        ax.text(
            pix[:, 0].min(),
            pix[:, 1].min() - 3,
            label,
            color=color,
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.4, pad=1),
        )


def select_predictions(
    preds: dict, num_classes: int, threshold: float, topk: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj_logits = preds["obj_logits"].reshape(-1)
    cls_logits = preds["cls_logits"].reshape(-1, num_classes)
    box_preds = preds["box_preds"].reshape(-1, 7)

    obj_scores = torch.sigmoid(obj_logits)
    cls_probs = torch.softmax(cls_logits, dim=1)
    cls_scores, cls_labels = cls_probs.max(dim=1)
    scores = obj_scores * cls_scores

    if scores.numel() == 0:
        return (
            torch.empty((0, 7)),
            torch.empty((0,)),
            torch.empty((0,), dtype=torch.long),
        )

    k = min(topk, scores.numel())
    top_scores, top_idx = torch.topk(scores, k=k)
    keep = top_scores >= threshold
    top_idx = top_idx[keep]
    return box_preds[top_idx].cpu(), top_scores[keep].cpu(), cls_labels[top_idx].cpu()


def render_sample(
    sample,
    cfg: TrainingConfig,
    output_dir: Path,
    idx: int,
    preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_boxes, pred_scores, pred_labels = preds if preds is not None else (None, None, None)
    cam_count = len(cfg.dataset.cameras)
    cols = min(3, cam_count)
    rows = math.ceil(cam_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for cam_idx, cam_name in enumerate(cfg.dataset.cameras):
        ax = axes[cam_idx]
        img = unnormalize(sample.images[cam_idx])
        intr = sample.intrinsics[cam_idx]
        extr = sample.extrinsics[cam_idx]
        h, w, _ = img.shape

        ax.imshow(img)
        ax.set_axis_off()

        # Ground truth boxes
        for box in sample.gt_boxes:
            pix = project_box_to_image(box, intr, extr, (w, h))
            if pix is None:
                continue
            draw_box(ax, pix, color="lime", label="GT")

        # Predictions
        if pred_boxes is not None:
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                pix = project_box_to_image(box, intr, extr, (w, h))
                if pix is None:
                    continue
                draw_box(ax, pix, color="red", label=f"P {int(label)} {score:.2f}")

        ax.set_title(f"{cam_name}")
        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)

    # Hide unused axes if any
    for extra_ax in axes[cam_count:]:
        extra_ax.set_visible(False)

    fig.suptitle(f"Sample {idx}", fontsize=14)
    fig.tight_layout()
    out_path = output_dir / f"sample_{idx:04d}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig.from_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = build_dataset(cfg, args.split)
    indices: Sequence[int] = args.indices if args.indices is not None else list(
        range(min(args.num_samples, len(dataset)))
    )

    model = None
    if args.model_checkpoint:
        model = load_model(cfg, args.model_checkpoint, device)
        print(f"Loaded model from {args.model_checkpoint}")

    for idx in indices:
        sample = dataset[idx]
        preds = None
        if model is not None:
            with torch.no_grad():
                batch_images = sample.images.unsqueeze(0).to(device)
                outputs = model(batch_images)
            preds = select_predictions(
                outputs, cfg.model.num_classes, threshold=args.obj_threshold, topk=args.topk
            )
        render_sample(sample, cfg, Path(args.output_dir), idx, preds)
        print(f"Saved visualization for sample {idx} to {args.output_dir}")


if __name__ == "__main__":
    main()
