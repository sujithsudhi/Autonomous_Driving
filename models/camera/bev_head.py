import torch
import torch.nn as nn


class BEVDetectionHead(nn.Module):
    def __init__(
        self, embed_dim: int, num_classes: int, bbox_dim: int = 7, dropout: float = 0.0
    ) -> None:
        super().__init__()
        cls_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        obj_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        box_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            cls_dropout,
            nn.Linear(embed_dim, embed_dim),
            final_dropout,
            nn.Linear(embed_dim, num_classes),
        )
        self.obj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            obj_dropout,
            nn.Linear(embed_dim, embed_dim // 2),
            final_dropout,
            nn.Linear(embed_dim // 2, 1),
        )
        self.box_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            box_dropout,
            nn.Linear(embed_dim, embed_dim),
            final_dropout,
            nn.Linear(embed_dim, bbox_dim),
        )

    def forward(self, bev_feats: torch.Tensor) -> torch.Tensor:
        cls_logits = self.cls_head(bev_feats)
        obj_logits = self.obj_head(bev_feats)
        boxes = self.box_head(bev_feats)
        return cls_logits, obj_logits, boxes

    def init_obj_bias(self, bias_value: float = -2.0) -> None:
        """Optionally initialize objectness logits toward 'no object'."""
        final_linear = self.obj_head[-1]
        if hasattr(final_linear, "bias") and final_linear.bias is not None:
            nn.init.constant_(final_linear.bias, bias_value)

    def init_obj_bias_from_prob(self, prior_prob: float = 0.01) -> None:
        """Set objectness bias from a desired prior probability."""
        prior_prob = max(min(prior_prob, 1 - 1e-4), 1e-4)
        bias = torch.log(torch.tensor(prior_prob) / torch.tensor(1.0 - prior_prob))
        self.init_obj_bias(float(bias))
