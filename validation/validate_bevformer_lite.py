import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
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
        "--iou-thresh",
        type=float,
        default=0.25,
        help="IoU threshold used when matching predictions to ground truth for metrics.",
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
    obj_logits: torch.Tensor,
    box_preds: torch.Tensor,
    score_thresh: float,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls_probs = torch.softmax(cls_logits, dim=-1)
    obj_probs = torch.sigmoid(obj_logits).squeeze(-1)
    class_scores, labels = cls_probs.max(dim=-1)
    scores = class_scores * obj_probs
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


def _cross_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def _edge_function(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> float:
    return _cross_2d(edge_end - edge_start, point - edge_start)


def _segment_intersection(
    p1: np.ndarray, p2: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray
) -> np.ndarray:
    r = p2 - p1
    s = edge_end - edge_start
    denom = _cross_2d(r, s)
    if abs(denom) < 1e-8:
        return (p1 + p2) / 2.0
    t = _cross_2d(edge_start - p1, s) / denom
    return p1 + t * r


def _polygon_clip(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    output = subject_polygon
    for idx in range(len(clip_polygon)):
        clip_start = clip_polygon[idx]
        clip_end = clip_polygon[(idx + 1) % len(clip_polygon)]
        input_list = output
        if len(input_list) == 0:
            break
        output_list: List[np.ndarray] = []
        start_point = input_list[-1]
        for end_point in input_list:
            if _edge_function(end_point, clip_start, clip_end) >= 0:
                if _edge_function(start_point, clip_start, clip_end) < 0:
                    output_list.append(_segment_intersection(start_point, end_point, clip_start, clip_end))
                output_list.append(end_point)
            elif _edge_function(start_point, clip_start, clip_end) >= 0:
                output_list.append(_segment_intersection(start_point, end_point, clip_start, clip_end))
            start_point = end_point
        output = np.array(output_list, dtype=np.float32)
    return output


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def bev_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    poly_a = _box_corners_xy(box_a)
    poly_b = _box_corners_xy(box_b)
    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)
    inter_poly = _polygon_clip(poly_a, poly_b)
    inter_area = _polygon_area(inter_poly)
    union = area_a + area_b - inter_area
    if union <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, inter_area / union))


def match_predictions_to_gt(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float
) -> List[Tuple[int, int, float]]:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for p_idx, pred in enumerate(pred_boxes):
        for g_idx, gt in enumerate(gt_boxes):
            ious[p_idx, g_idx] = bev_box_iou(pred, gt)

    matches: List[Tuple[int, int, float]] = []
    while True:
        max_idx = np.unravel_index(np.argmax(ious), ious.shape)
        max_iou = ious[max_idx]
        if max_iou < iou_threshold or max_iou <= 0:
            break
        p_idx, g_idx = max_idx
        matches.append((p_idx, g_idx, float(max_iou)))
        ious[p_idx, :] = -1.0
        ious[:, g_idx] = -1.0
    return matches


class MetricTracker:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = max(1, num_classes)
        self.class_correct = np.zeros(self.num_classes, dtype=np.int64)
        self.class_total = np.zeros(self.num_classes, dtype=np.int64)
        self.iou_sums = np.zeros(self.num_classes, dtype=np.float64)
        self.iou_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.total_correct = 0
        self.total_gt = 0
        self.total_iou = 0.0
        self.total_iou_count = 0

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        iou_threshold: float,
    ) -> None:
        if gt_labels is None or len(gt_labels) == 0:
            return

        matches = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold)
        gt_to_match = {g_idx: (p_idx, iou) for p_idx, g_idx, iou in matches}

        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_label < 0 or gt_label >= self.num_classes:
                continue
            self.class_total[gt_label] += 1
            self.total_gt += 1
            self.iou_counts[gt_label] += 1
            self.total_iou_count += 1
            match = gt_to_match.get(gt_idx)
            if match is None:
                continue
            pred_idx, iou = match
            self.iou_sums[gt_label] += iou
            self.total_iou += iou
            pred_label = int(pred_labels[pred_idx]) if pred_idx < len(pred_labels) else -1
            if 0 <= pred_label < self.num_classes and pred_label == gt_label:
                self.class_correct[gt_label] += 1
                self.total_correct += 1

    def summary(self, class_names: Sequence[str]) -> Dict[str, object]:
        names = [name if name else f"class_{idx}" for idx, name in enumerate(class_names)]
        per_class: List[Dict[str, object]] = []
        for idx in range(self.num_classes):
            total = int(self.class_total[idx])
            iou_count = int(self.iou_counts[idx])
            acc = (self.class_correct[idx] / total) if total > 0 else None
            mean_iou = (self.iou_sums[idx] / max(iou_count, 1)) if iou_count > 0 else None
            per_class.append(
                {
                    "class": names[idx] if idx < len(names) else f"class_{idx}",
                    "samples": total,
                    "accuracy": None if acc is None else float(acc),
                    "mean_iou": None if mean_iou is None else float(mean_iou),
                }
            )

        overall_acc = self.total_correct / self.total_gt if self.total_gt > 0 else None
        overall_iou = self.total_iou / self.total_iou_count if self.total_iou_count > 0 else None
        return {
            "overall": {
                "accuracy": None if overall_acc is None else float(overall_acc),
                "mean_iou": None if overall_iou is None else float(overall_iou),
                "samples": int(self.total_gt),
            },
            "per_class": per_class,
        }


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
    gt_face_color = (0.2, 0.8, 0.2, 0.1)
    gt_edge_color = (0.1, 0.6, 0.1, 0.9)
    pred_edge_color = (0.85, 0.1, 0.1, 0.95)

    if gt_boxes is not None and gt_labels is not None and len(gt_boxes) > 0:
        for box_idx, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            corners = _box_corners_xy(box)
            poly = Polygon(
                corners,
                closed=True,
                fill=True,
                facecolor=gt_face_color,
                edgecolor=gt_edge_color,
                linewidth=1.8,
                linestyle=":",
                alpha=0.9,
                label="Ground Truth" if box_idx == 0 else None,
            )
            ax.add_patch(poly)
            text = class_names[label] if label < len(class_names) and class_names[label] else str(label)
            ax.text(
                box[0],
                box[1] - 0.8,
                f"GT:{text}",
                color="green",
                fontsize=8,
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.2),
            )

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        corners = _box_corners_xy(box)
        poly = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=pred_edge_color,
            linewidth=2.0,
            alpha=0.95,
            linestyle="-",
            label="Predictions" if idx == 0 else None,
        )
        ax.add_patch(poly)
        text = class_names[label] if label < len(class_names) and class_names[label] else str(label)
        ax.text(
            box[0],
            box[1] + 0.8,
            f"Pred:{text} ({score:.2f})",
            color="red",
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.2),
        )

    handles, labels_text = ax.get_legend_handles_labels()
    custom_handles = []
    if "Ground Truth" not in labels_text:
        custom_handles.append(
            Line2D([0], [0], color=gt_edge_color, linestyle=":", linewidth=2, label="Ground Truth")
        )
    if "Predictions" not in labels_text:
        custom_handles.append(
            Line2D([0], [0], color=pred_edge_color, linestyle="-", linewidth=2, label="Predictions")
        )
    handles.extend(custom_handles)
    labels_text.extend([h.get_label() for h in custom_handles])
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
    metrics: Optional[MetricTracker] = None,
    iou_threshold: float = 0.25,
) -> None:
    images = batch["images"].to(device, non_blocking=True)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            preds = model(images)

    boxes, labels, scores = decode_predictions(
        preds["cls_logits"].cpu(),
        preds["obj_logits"].cpu(),
        preds["box_preds"].cpu(),
        score_thresh,
        topk,
    )
    gt_mask = batch["gt_masks"][0] > 0.5
    gt_boxes = batch["gt_boxes"][0][gt_mask].cpu().numpy()
    gt_labels = batch["gt_labels"][0][gt_mask].cpu().numpy()

    if metrics is not None:
        metrics.update(boxes, labels, gt_boxes, gt_labels, iou_threshold)

    _write_prediction_report(
        save_path.with_suffix(".txt"),
        boxes,
        labels,
        scores,
        gt_boxes,
        gt_labels,
        class_names,
    )

    draw_bev(boxes, labels, scores, bev_bounds, class_names, save_path, gt_boxes, gt_labels)


def _write_prediction_report(
    path: Path,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    class_names: Sequence[str],
) -> None:
    def _class_name(idx: int) -> str:
        if 0 <= idx < len(class_names) and class_names[idx]:
            return class_names[idx]
        return f"class_{idx}"

    def _format_box(box: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(v):.3f}" for v in box.tolist()) + "]"

    lines: List[str] = []
    lines.append("Predictions:")
    if len(pred_boxes) == 0:
        lines.append("  (none)")
    else:
        for idx, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores), start=1):
            lines.append(
                f"  {idx:02d}. class={_class_name(int(label))} score={float(score):.3f} box={_format_box(box)}"
            )

    lines.append("\nGround Truth:")
    if gt_boxes is None or len(gt_boxes) == 0:
        lines.append("  (none)")
    else:
        for idx, (box, label) in enumerate(zip(gt_boxes, gt_labels), start=1):
            lines.append(f"  {idx:02d}. class={_class_name(int(label))} box={_format_box(box)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_bevformer,
    )

    model = build_model(cfg, args.checkpoint, device)
    class_names = invert_category_mapping(CATEGORY_MAPPING)
    metrics = MetricTracker(len(class_names))

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
            metrics=metrics,
            iou_threshold=args.iou_thresh,
        )
        print(f"Saved visualization to {save_path}")

    metrics_summary = metrics.summary(class_names)
    metrics_path = output_dir / f"{args.split}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, indent=2)

    print("\n=== Validation Metrics ===")
    overall = metrics_summary["overall"]
    if overall["accuracy"] is not None:
        print(f"Overall class accuracy: {overall['accuracy']:.4f} across {overall['samples']} GT boxes")
    else:
        print("Overall class accuracy: N/A")
    if overall["mean_iou"] is not None:
        print(f"Overall mean IoU: {overall['mean_iou']:.4f}")
    else:
        print("Overall mean IoU: N/A")

    print("\nPer-class metrics:")
    for entry in metrics_summary["per_class"]:
        name = entry["class"]
        samples = entry["samples"]
        acc = entry["accuracy"]
        miou = entry["mean_iou"]
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        miou_str = f"{miou:.4f}" if miou is not None else "N/A"
        print(f"- {name:>16s}: samples={samples:4d} | acc={acc_str} | IoU={miou_str}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
