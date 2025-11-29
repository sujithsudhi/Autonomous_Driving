"""Quick sanity check that GT boxes project inside camera images."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_datasets.nuscenes_bev import NuScenesBEVDataset  # noqa: E402
from training.config import TrainingConfig  # noqa: E402
from utils.geometry import project_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GT box projection coverage.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bevformer_lite_nuscenes.yaml",
        help="Training config path.",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to scan (starting from index 0).",
    )
    return parser.parse_args()


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


def project_box_inside(
    box: torch.Tensor, intr: torch.Tensor, extr: torch.Tensor, image_size: Tuple[int, int]
) -> bool:
    corners = box_corners(box)
    cam_space = (extr @ torch.cat([corners, torch.ones_like(corners[:, :1])], dim=-1).T).T[:, 2]
    if (cam_space > 0.1).sum() < 8:
        return False
    pix = project_points(corners, intr, extr)
    w, h = image_size
    inside = (
        (pix[:, 0] >= 0)
        & (pix[:, 0] <= w - 1)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] <= h - 1)
    )
    return bool(inside.all())


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig.from_yaml(args.config)
    dataset = NuScenesBEVDataset(
        dataroot=cfg.dataset.root,
        version=cfg.dataset.version,
        cameras=cfg.dataset.cameras,
        bev_bounds=cfg.dataset.bev_bounds,
        split=args.split,
        image_size=cfg.dataset.image_size,
    )

    total = {cam: 0 for cam in cfg.dataset.cameras}
    inside = {cam: 0 for cam in cfg.dataset.cameras}

    scan_n = min(args.num_samples, len(dataset))
    for idx in range(scan_n):
        sample = dataset[idx]
        _, h, w = sample.images[0].shape  # images are normalized; shape (C,H,W)
        for cam_idx, cam_name in enumerate(cfg.dataset.cameras):
            intr = sample.intrinsics[cam_idx]
            extr = sample.extrinsics[cam_idx]
            for box in sample.gt_boxes:
                total[cam_name] += 1
                if project_box_inside(box, intr, extr, (w, h)):
                    inside[cam_name] += 1

    print(f"Scanned {scan_n} samples")
    for cam in cfg.dataset.cameras:
        if total[cam] == 0:
            print(f"{cam}: no GT boxes seen")
            continue
        frac = 100.0 * inside[cam] / max(total[cam], 1)
        print(f"{cam}: {inside[cam]}/{total[cam]} boxes inside image ({frac:.1f}%)")


if __name__ == "__main__":
    main()
