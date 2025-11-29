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
        obj_weight: float = 1.0,
        obj_pos_weight: float = 1.0,
        neg_pos_ratio: float = 3.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.bev_bounds = bev_bounds
        self.bev_resolution = bev_resolution
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.obj_pos_weight = obj_pos_weight
        self.neg_pos_ratio = neg_pos_ratio
        self.label_smoothing = label_smoothing
        self._warned_empty = False
        self._seen_xy_min = None
        self._seen_xy_max = None

        x_min, x_max = bev_bounds["x"]
        y_min, y_max = bev_bounds["y"]
        self.bev_w = int((x_max - x_min) / bev_resolution)
        self.bev_h = int((y_max - y_min) / bev_resolution)

    def forward(
        self,
        cls_logits: torch.Tensor,
        obj_logits: torch.Tensor,
        box_preds: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz = cls_logits.shape[0]
        num_cells = self.bev_w * self.bev_h
        cls_logits = cls_logits.view(bsz, num_cells, -1)
        obj_logits = obj_logits.view(bsz, num_cells)
        box_preds = box_preds.view(bsz, num_cells, -1)

        device = cls_logits.device
        cls_targets = torch.full(
            (bsz, num_cells), fill_value=-1, device=device, dtype=torch.long
        )
        box_targets = torch.zeros_like(box_preds)
        positive_mask = torch.zeros((bsz, num_cells), device=device, dtype=torch.bool)

        for b in range(bsz):
            boxes = gt_boxes[b][gt_masks[b] > 0.5]
            labels = gt_labels[b][gt_masks[b] > 0.5]

            if boxes.numel() > 0:
                self._update_seen_ranges(boxes)
            for box, label in zip(boxes, labels):
                if label < 0:
                    continue
                idx = self._box_to_index(box)
                if idx is None or positive_mask[b, idx]:
                    continue
                cls_targets[b, idx] = int(label.item())
                box_targets[b, idx] = box
                positive_mask[b, idx] = True

        positive = int(positive_mask.sum().item())
        if positive == 0 and not self._warned_empty:
            print(
                "[BEVDetectionLoss] No ground-truth boxes fell inside the BEV grid; "
                "objectness/box regression losses will stay near zero. "
                "Check bev_bounds relative to the dataset."
            )
            if self._seen_xy_min is not None and self._seen_xy_max is not None:
                x_min, x_max = self.bev_bounds["x"]
                y_min, y_max = self.bev_bounds["y"]
                print(
                    f"[BEVDetectionLoss] Observed GT x range "
                    f"[{self._seen_xy_min[0]:.2f}, {self._seen_xy_max[0]:.2f}], "
                    f"y range [{self._seen_xy_min[1]:.2f}, {self._seen_xy_max[1]:.2f}] "
                    f"vs bev_bounds x=({x_min:.2f},{x_max:.2f}) y=({y_min:.2f},{y_max:.2f})."
                )
            self._warned_empty = True

        pos_logits = obj_logits[positive_mask]
        if pos_logits.numel() > 0:
            obj_pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="sum"
            ) / max(pos_logits.numel(), 1)
        else:
            obj_pos_loss = torch.zeros(1, device=device)

        neg_mask = ~positive_mask
        neg_logits = obj_logits[neg_mask]
        if neg_logits.numel() > 0:
            if self.neg_pos_ratio > 0 and positive > 0:
                max_neg = int(self.neg_pos_ratio * max(positive, 1))
                max_neg = max(1, max_neg)
                max_neg = min(max_neg, neg_logits.numel())
                if 0 < max_neg < neg_logits.numel():
                    neg_probs = torch.sigmoid(neg_logits)
                    topk_idx = torch.topk(neg_probs, k=max_neg).indices
                    neg_logits = neg_logits[topk_idx]
            obj_neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="sum"
            ) / max(neg_logits.numel(), 1)
        else:
            obj_neg_loss = torch.zeros(1, device=device)

        obj_loss = obj_pos_loss + obj_neg_loss

        if positive > 0:
            cls_loss = F.cross_entropy(
                cls_logits[positive_mask],
                cls_targets[positive_mask],
                label_smoothing=self.label_smoothing,
            )
            box_loss = F.smooth_l1_loss(box_preds[positive_mask], box_targets[positive_mask])
        else:
            cls_loss = torch.zeros(1, device=device)
            box_loss = torch.zeros(1, device=device)

        total = (
            self.obj_weight * (self.obj_pos_weight * obj_pos_loss + obj_neg_loss)
            + self.cls_weight * cls_loss
            + self.box_weight * box_loss
        )
        return {
            "loss_total": total,
            "loss_cls": cls_loss,
            "loss_box": box_loss,
            "loss_obj": obj_loss,
            "num_positive": torch.tensor(float(positive), device=device),
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

    def _update_seen_ranges(self, boxes: torch.Tensor) -> None:
        # Track min/max xy of incoming ground truth boxes to help debug bounds.
        xy = boxes[:, :2]
        batch_min = xy.min(dim=0).values
        batch_max = xy.max(dim=0).values
        if self._seen_xy_min is None:
            self._seen_xy_min = batch_min.detach().cpu()
            self._seen_xy_max = batch_max.detach().cpu()
        else:
            self._seen_xy_min = torch.minimum(self._seen_xy_min, batch_min.detach().cpu())
            self._seen_xy_max = torch.maximum(self._seen_xy_max, batch_max.detach().cpu())
