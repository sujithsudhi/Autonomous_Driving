import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.camera.bevformer_lite import BEVFormerLite
from train_datasets.nuscenes_bev import NuScenesBEVDataset, collate_bevformer
from utils.nuscenes_utils import CATEGORY_MAPPING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BEVFormer-Lite checkpoints")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bevformer_lite_nuscenes.yaml",
        help="Path to training config used for the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a saved checkpoint containing the model state dict.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation/outputs",
        help="Where to save BEV visualizations.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "mini_train", "mini_val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Number of samples to visualize before stopping.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.35,
        help="Score threshold for predicted boxes.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=75,
        help="Maximum number of predictions to draw per frame after thresholding.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Data loader workers used during validation.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(cfg: Dict, checkpoint_path: str, device: torch.device) -> BEVFormerLite:
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

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def invert_category_mapping(mapping: Dict[str, int]) -> List[str]:
    names: List[str] = [""] * (max(mapping.values()) + 1)
    for name, idx in mapping.items():
        if idx < len(names):
            names[idx] = name
    return names


def decode_predictions(
    cls_logits: torch.Tensor,
    box_preds: torch.Tensor,
    score_thresh: float,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = torch.sigmoid(cls_logits)
    scores, labels = probs.max(dim=-1)
    flat_scores = scores.reshape(-1)
    flat_labels = labels.reshape(-1)
    flat_boxes = box_preds.reshape(-1, box_preds.shape[-1])

    keep = flat_scores > score_thresh
    if keep.sum() == 0:
        return (
            np.zeros((0, box_preds.shape[-1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    flat_scores = flat_scores[keep]
    flat_labels = flat_labels[keep]
    flat_boxes = flat_boxes[keep]

    if flat_scores.numel() > topk:
        topk_idx = torch.topk(flat_scores, topk).indices
        flat_scores = flat_scores[topk_idx]
        flat_labels = flat_labels[topk_idx]
        flat_boxes = flat_boxes[topk_idx]

    return (
        flat_boxes.cpu().numpy(),
        flat_labels.cpu().numpy(),
        flat_scores.cpu().numpy(),
    )


def _box_corners_xy(box: np.ndarray) -> np.ndarray:
    cx, cy, _cz, width, length, _height, yaw = box
    half_l = length / 2.0
    half_w = width / 2.0
    rot = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32
    )
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    return corners @ rot.T + np.array([cx, cy])


def draw_bev(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    bev_bounds: Dict[str, Sequence[float]],
    class_names: Sequence[str],
    save_path: Path,
    gt_boxes: Optional[np.ndarray] = None,
    gt_labels: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("BEV Predictions")
    ax.set_xlim(bev_bounds["x"][0], bev_bounds["x"][1])
    ax.set_ylim(bev_bounds["y"][0], bev_bounds["y"][1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    colors = plt.cm.get_cmap("tab20", max(1, len(class_names)))

    if gt_boxes is not None and gt_labels is not None and len(gt_boxes) > 0:
        for box, label in zip(gt_boxes, gt_labels):
            corners = _box_corners_xy(box)
            poly = Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor="green",
                linewidth=1.5,
                linestyle=":",
                alpha=0.7,
                label="GT" if label == gt_labels[0] else None,
            )
            ax.add_patch(poly)

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        corners = _box_corners_xy(box)
        poly = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=colors(label % colors.N),
            linewidth=2.0,
            alpha=0.85,
        )
        ax.add_patch(poly)
        text = class_names[label] if label < len(class_names) else str(label)
        ax.text(
            box[0],
            box[1],
            f"{text}:{score:.2f}",
            color=colors(label % colors.N),
            fontsize=8,
            ha="center",
        )

    handles, labels_text = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_text, loc="upper right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_inference(
    model: BEVFormerLite,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    score_thresh: float,
    topk: int,
    bev_bounds: Dict[str, Sequence[float]],
    class_names: Sequence[str],
    save_path: Path,
) -> None:
    images = batch["images"].to(device)

    with torch.no_grad():
        preds = model(images)

    boxes, labels, scores = decode_predictions(
        preds["cls_logits"].cpu(), preds["box_preds"].cpu(), score_thresh, topk
    )
    gt_mask = batch["gt_masks"][0] > 0.5
    gt_boxes = batch["gt_boxes"][0][gt_mask].cpu().numpy()
    gt_labels = batch["gt_labels"][0][gt_mask].cpu().numpy()

    draw_bev(boxes, labels, scores, bev_bounds, class_names, save_path, gt_boxes, gt_labels)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NuScenesBEVDataset(
        dataroot=cfg["dataset"]["root"],
        version=cfg["dataset"]["version"],
        cameras=cfg["dataset"]["cameras"],
        bev_bounds=cfg["dataset"]["bev_bounds"],
        split=args.split,
        image_size=cfg["dataset"].get("image_size"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bevformer,
    )

    model = build_model(cfg, args.checkpoint, device)
    class_names = invert_category_mapping(CATEGORY_MAPPING)

    output_dir = Path(args.output_dir)
    for idx, batch in enumerate(dataloader):
        if idx >= args.max_samples:
            break
        save_path = output_dir / f"{args.split}_sample_{idx:04d}.png"
        run_inference(
            model,
            batch,
            device,
            args.score_thresh,
            args.topk,
            cfg["dataset"]["bev_bounds"],
            class_names,
            save_path,
        )
        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
