import torch
import torch.nn as nn


class BEVDetectionHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, bbox_dim: int = 7) -> None:
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )
        self.obj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.box_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, bbox_dim),
        )

    def forward(self, bev_feats: torch.Tensor) -> torch.Tensor:
        cls_logits = self.cls_head(bev_feats)
        obj_logits = self.obj_head(bev_feats)
        boxes = self.box_head(bev_feats)
        return cls_logits, obj_logits, boxes
