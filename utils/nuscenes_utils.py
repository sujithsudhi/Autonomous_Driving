from typing import Dict, Tuple

import numpy as np

try:
    from nuscenes.nuscenes import NuScenes
except ImportError as err:  # pragma: no cover - nuScenes not installed on CI
    NuScenes = None  # type: ignore
    _NUSCENES_IMPORT_ERROR = err
else:
    _NUSCENES_IMPORT_ERROR = None

from utils.geometry import transform_matrix


CATEGORY_MAPPING = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "trailer": 3,
    "construction_vehicle": 4,
    "pedestrian": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic_cone": 8,
    "barrier": 9,
}


def get_nuscenes_handle(version: str, dataroot: str) -> "NuScenes":
    if NuScenes is None:
        raise RuntimeError(
            "nuScenes devkit is not installed. Please install nuscenes-devkit to continue."
        ) from _NUSCENES_IMPORT_ERROR
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def load_calibration_matrices(nusc: "NuScenes", sample_data_rec: Dict) -> Tuple[np.ndarray, np.ndarray]:
    cs_rec = nusc.get("calibrated_sensor", sample_data_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sample_data_rec["ego_pose_token"])
    cam_intr = np.array(cs_rec["camera_intrinsic"], dtype=np.float32)
    # From camera to ego then to global
    cam_to_ego = transform_matrix(cs_rec["translation"], cs_rec["rotation"])
    ego_to_global = transform_matrix(pose_rec["translation"], pose_rec["rotation"])
    cam_to_global = ego_to_global @ cam_to_ego
    extrinsic = np.linalg.inv(cam_to_global)
    return cam_intr, extrinsic.astype(np.float32)


def annotation_to_array(ann_rec: Dict) -> Tuple[np.ndarray, int]:
    center = np.array(ann_rec["translation"], dtype=np.float32)
    size = np.array(ann_rec["size"], dtype=np.float32)[[1, 0, 2]]  # NuScenes uses lwh
    yaw = ann_rec["rotation"]
    quat = np.array(yaw, dtype=np.float32)
    heading = 2 * np.arctan2(quat[3], quat[0])
    box = np.concatenate([center, size, np.array([heading], dtype=np.float32)], axis=0)
    category = ann_rec["category_name"].split(".")[0]
    label = CATEGORY_MAPPING.get(category, -1)
    return box, label
