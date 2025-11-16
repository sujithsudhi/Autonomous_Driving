from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def quaternion_to_matrix(rotation: Iterable[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] or [x, y, z, w] to rotation matrix."""
    q = np.array(rotation, dtype=np.float64).flatten()
    if q.shape[0] != 4:
        raise ValueError("Quaternion must have four elements.")
    if abs(np.linalg.norm(q) - 1.0) > 1e-5:
        q = q / (np.linalg.norm(q) + 1e-8)
    if abs(q[0]) < abs(q[-1]):
        w, x, y, z = q[-1], q[0], q[1], q[2]
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return rot


def transform_matrix(
    translation: Iterable[float],
    rotation: Iterable[float],
    inverse: bool = False,
) -> np.ndarray:
    """Create a homogeneous transform matrix from translation + quaternion."""
    rot = quaternion_to_matrix(rotation)
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = np.array(translation)
    if inverse:
        rot_t = rot.T
        inv = np.eye(4)
        inv[:3, :3] = rot_t
        inv[:3, 3] = -rot_t @ np.array(translation)
        return inv
    return mat


def project_points(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points in ego frame into image pixel coordinates."""
    if points.shape[-1] != 3:
        raise ValueError("points must be (N,3)")
    ones = torch.ones_like(points[..., :1])
    hom = torch.cat([points, ones], dim=-1)  # (...,4)
    cam_points = (extrinsics @ hom.unsqueeze(-1)).squeeze(-1)[..., :3]
    pix = cam_points @ intrinsics.transpose(-1, -2)
    pix_xy = pix[..., :2] / torch.clamp(pix[..., 2:3], min=1e-5)
    return pix_xy


def create_bev_grid(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    resolution: float,
    device: torch.device,
) -> torch.Tensor:
    """Return BEV query centers on XY plane."""
    xs = torch.arange(x_bounds[0], x_bounds[1], resolution, device=device)
    ys = torch.arange(y_bounds[0], y_bounds[1], resolution, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    zeros = torch.zeros_like(grid_x)
    points = torch.stack([grid_x, grid_y, zeros], dim=-1)  # (H, W, 3)
    return points.view(-1, 3)


def compute_frustum_mask(
    bev_points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Mark BEV points that fall inside camera frustum."""
    proj = project_points(bev_points, intrinsics, extrinsics)
    width, height = image_size
    in_front = bev_points[:, 0] > 1e-2
    inside = (
        (proj[:, 0] >= 0)
        & (proj[:, 0] <= width - 1)
        & (proj[:, 1] >= 0)
        & (proj[:, 1] <= height - 1)
    )
    return in_front & inside


def normalize_boxes(boxes: torch.Tensor, bev_bounds: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """Normalize box centers to [-1,1] range for transformer positional encodings."""
    x_min, x_max = bev_bounds["x"]
    y_min, y_max = bev_bounds["y"]
    z_min, z_max = bev_bounds["z"]
    centers = boxes[..., :3]
    sizes = boxes[..., 3:6]
    yaw = boxes[..., 6:7]

    centers_norm = centers.clone()
    centers_norm[..., 0] = 2 * (centers[..., 0] - x_min) / (x_max - x_min) - 1
    centers_norm[..., 1] = 2 * (centers[..., 1] - y_min) / (y_max - y_min) - 1
    centers_norm[..., 2] = 2 * (centers[..., 2] - z_min) / (z_max - z_min) - 1
    normalized = torch.cat([centers_norm, sizes, yaw], dim=-1)
    return normalized
