import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.geometry import quaternion_to_matrix, transform_matrix
from utils.nuscenes_utils import (
    annotation_to_array,
    get_nuscenes_handle,
    load_calibration_matrices,
)
from nuscenes.utils.splits import create_splits_scenes


@dataclass
class Sample:
    images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    gt_boxes: torch.Tensor
    gt_classes: torch.Tensor


class NuScenesBEVDataset(Dataset):
    """Camera-only nuScenes dataset tailored for BEVFormer-Lite."""

    def __init__(
        self,
        dataroot: str,
        version: str,
        cameras: Sequence[str],
        bev_bounds: Dict[str, Sequence[float]],
        transform: Optional[Any] = None,
        split: str = "train",
        image_size: Optional[Sequence[int]] = None,
    ) -> None:
        self.dataroot = dataroot
        self.version = version
        self.cameras = list(cameras)
        self.transform = transform or self._default_transform()
        self.split = split
        self.image_size = tuple(image_size) if image_size is not None else None
        self.nusc = get_nuscenes_handle(version=version, dataroot=dataroot)

        self.sample_tokens = self._collect_split_tokens(split)
        self.bev_bounds = bev_bounds

    def _collect_split_tokens(self, split: str) -> List[str]:
        split_file = os.path.join(
            self.dataroot, "splits", self.version, f"{split}.txt"
        )
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        # Fall back to NuScenes built-in splits when custom list not provided.
        scene_splits = create_splits_scenes()
        scenes = set(scene_splits.get(split, []))
        tokens: List[str] = []
        if scenes:
            for scene_rec in self.nusc.scene:
                if scene_rec["name"] not in scenes:
                    continue
                sample_token = scene_rec["first_sample_token"]
                while sample_token:
                    tokens.append(sample_token)
                    sample_rec = self.nusc.get("sample", sample_token)
                    sample_token = sample_rec["next"]
        return tokens

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _load_images(self, sample_rec: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = []
        intrinsics = []
        extrinsics = []
        for cam_name in self.cameras:
            data_token = sample_rec["data"][cam_name]
            sd_rec = self.nusc.get("sample_data", data_token)
            im = Image.open(os.path.join(self.nusc.dataroot, sd_rec["filename"])).convert("RGB")
            orig_w, orig_h = im.size
            if self.image_size is not None:
                im = im.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            im = self.transform(im)

            cam_intr, cam_extr = load_calibration_matrices(self.nusc, sd_rec)
            if self.image_size is not None and (orig_w != self.image_size[1] or orig_h != self.image_size[0]):
                scale_w = self.image_size[1] / float(orig_w)
                scale_h = self.image_size[0] / float(orig_h)
                cam_intr = cam_intr.copy()
                cam_intr[0, :] *= scale_w
                cam_intr[1, :] *= scale_h
            images.append(im)
            intrinsics.append(torch.from_numpy(cam_intr).float())
            extrinsics.append(torch.from_numpy(cam_extr).float())

        return (
            torch.stack(images, dim=0),
            torch.stack(intrinsics, dim=0),
            torch.stack(extrinsics, dim=0),
        )

    def _get_world_to_ego(self, sample_rec: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Return world-to-ego transform and ego yaw for current sample."""
        ref_sensor = self.cameras[0]
        ref_token = sample_rec["data"][ref_sensor]
        sd_rec = self.nusc.get("sample_data", ref_token)
        ego_pose = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        world_to_ego = transform_matrix(ego_pose["translation"], ego_pose["rotation"], inverse=True)
        rot = quaternion_to_matrix(ego_pose["rotation"])
        ego_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        return world_to_ego.astype(np.float32), ego_yaw

    def _load_boxes(self, sample_rec: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_boxes = []
        gt_labels = []
        world_to_ego, ego_yaw = self._get_world_to_ego(sample_rec)
        for ann_token in sample_rec["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            box, label = annotation_to_array(ann)
            center = np.concatenate([box[:3], np.array([1.0], dtype=np.float32)])
            center_ego = world_to_ego @ center
            box[:3] = center_ego[:3]
            box[6] = ((box[6] - ego_yaw + np.pi) % (2 * np.pi)) - np.pi
            gt_boxes.append(box)
            gt_labels.append(label)

        if not gt_boxes:
            gt_boxes = torch.zeros((0, 7), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.long)
        else:
            gt_boxes = torch.tensor(np.stack(gt_boxes), dtype=torch.float32)
            gt_labels = torch.tensor(np.stack(gt_labels), dtype=torch.long)
        return gt_boxes, gt_labels

    def __getitem__(self, idx: int) -> Sample:
        sample_token = self.sample_tokens[idx]
        sample_rec = self.nusc.get("sample", sample_token)
        images, intrinsics, extrinsics = self._load_images(sample_rec)
        gt_boxes, gt_labels = self._load_boxes(sample_rec)
        return Sample(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            gt_boxes=gt_boxes,
            gt_classes=gt_labels,
        )

    @staticmethod
    def _default_transform() -> T.Compose:
        """Basic RGB normalization matching ImageNet-pretrained backbones."""
        return T.Compose(
            [
                T.ToTensor(),  # scales to [0,1]
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def collate_bevformer(batch: Sequence[Sample]) -> Dict[str, torch.Tensor]:
    """Pad variable number of boxes while stacking image tensors."""
    images = torch.stack([sample.images for sample in batch], dim=0)
    intrinsics = torch.stack([sample.intrinsics for sample in batch], dim=0)
    extrinsics = torch.stack([sample.extrinsics for sample in batch], dim=0)

    max_boxes = max(sample.gt_boxes.shape[0] for sample in batch)
    padded_boxes = []
    padded_labels = []
    masks = []
    for sample in batch:
        count = sample.gt_boxes.shape[0]
        pad = max_boxes - count
        if pad > 0:
            boxes = torch.cat(
                [
                    sample.gt_boxes,
                    torch.zeros((pad, sample.gt_boxes.shape[1]), device=sample.gt_boxes.device),
                ],
                dim=0,
            )
            labels = torch.cat(
                [
                    sample.gt_classes,
                    torch.full((pad,), -1, dtype=torch.long, device=sample.gt_classes.device),
                ],
                dim=0,
            )
            mask = torch.cat([torch.ones(count), torch.zeros(pad)], dim=0)
        else:
            boxes = sample.gt_boxes
            labels = sample.gt_classes
            mask = torch.ones(count)
        padded_boxes.append(boxes)
        padded_labels.append(labels)
        masks.append(mask)

    return {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "gt_boxes": torch.stack(padded_boxes, dim=0),
        "gt_labels": torch.stack(padded_labels, dim=0),
        "gt_masks": torch.stack(masks, dim=0),
    }
