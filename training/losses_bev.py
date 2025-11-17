from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVDetectionLoss(nn.Module):
    def __init__(
        self,
        bev_bounds: Dict[str, Tuple[float, float]],
        bev_resolution: float,
        cls_weight: float = 1.0,
        box_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bev_bounds = bev_bounds
        self.bev_resolution = bev_resolution
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self._warned_empty = False

        x_min, x_max = bev_bounds["x"]
        y_min, y_max = bev_bounds["y"]
        self.bev_w = int((x_max - x_min) / bev_resolution)
        self.bev_h = int((y_max - y_min) / bev_resolution)

    def forward(
        self,
        cls_logits: torch.Tensor,
        box_preds: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz = cls_logits.shape[0]
        num_cells = self.bev_w * self.bev_h
        cls_logits = cls_logits.view(bsz, num_cells, -1)
        box_preds = box_preds.view(bsz, num_cells, -1)

        cls_targets = torch.zeros_like(cls_logits)
        box_targets = torch.zeros_like(box_preds)
        reg_masks = torch.zeros(bsz, num_cells, device=cls_logits.device)

        for b in range(bsz):
            boxes = gt_boxes[b][gt_masks[b] > 0.5]
            labels = gt_labels[b][gt_masks[b] > 0.5]
            for box, label in zip(boxes, labels):
                if label < 0:
                    continue
                idx = self._box_to_index(box)
                if idx is None:
                    continue
                cls_targets[b, idx, label] = 1.0
                box_targets[b, idx] = box
                reg_masks[b, idx] = 1.0

        positive = int(reg_masks.sum().item())
        if positive == 0 and not self._warned_empty:
            print(
                "[BEVDetectionLoss] No ground-truth boxes fell inside the BEV grid; "
                "box regression loss will stay at zero. Check bev_bounds relative to the dataset."
            )
            self._warned_empty = True

        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")
        cls_loss = cls_loss.sum(dim=-1).mean()

        if reg_masks.sum() > 0:
            box_loss = F.smooth_l1_loss(box_preds, box_targets, reduction="none").sum(-1)
            box_loss = (box_loss * reg_masks).sum() / (reg_masks.sum() + 1e-6)
        else:
            box_loss = torch.zeros(1, device=cls_logits.device)

        total = self.cls_weight * cls_loss + self.box_weight * box_loss
        return {
            "loss_total": total,
            "loss_cls": cls_loss,
            "loss_box": box_loss,
        }

    def _box_to_index(self, box: torch.Tensor) -> int:
        x, y = box[0], box[1]
        x_min, x_max = self.bev_bounds["x"]
        y_min, y_max = self.bev_bounds["y"]
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return None
        grid_x = int((x - x_min) / self.bev_resolution)
        grid_y = int((y - y_min) / self.bev_resolution)
        grid_x = min(max(grid_x, 0), self.bev_w - 1)
        grid_y = min(max(grid_y, 0), self.bev_h - 1)
        return grid_y * self.bev_w + grid_x
