from typing import Dict, Tuple

import numpy as np
from pyquaternion import Quaternion

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
    cam_intr = np.array(cs_rec["camera_intrinsic"], dtype=np.float32)

    # Build sensor->ego matrix using nuScenes convention (quaternion wxyz)
    rot = Quaternion(cs_rec["rotation"]).rotation_matrix
    trans = np.array(cs_rec["translation"], dtype=np.float32)
    sensor_to_ego = np.eye(4, dtype=np.float32)
    sensor_to_ego[:3, :3] = rot
    sensor_to_ego[:3, 3] = trans

    # We need ego->sensor for projecting ego-frame points into camera frame.
    ego_to_sensor = np.linalg.inv(sensor_to_ego)
    return cam_intr, ego_to_sensor.astype(np.float32)


def annotation_to_array(ann_rec: Dict) -> Tuple[np.ndarray, int]:
    center = np.array(ann_rec["translation"], dtype=np.float32)
    size = np.array(ann_rec["size"], dtype=np.float32)[[1, 0, 2]]  # NuScenes uses lwh
    yaw = ann_rec["rotation"]
    quat = np.array(yaw, dtype=np.float32)
    heading = 2 * np.arctan2(quat[3], quat[0])
    box = np.concatenate([center, size, np.array([heading], dtype=np.float32)], axis=0)
    label = -1
    for part in ann_rec["category_name"].split("."):
        if part in CATEGORY_MAPPING:
            label = CATEGORY_MAPPING[part]
            break
    return box, label
