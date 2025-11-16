from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.swin_backbone import SwinBackbone
from models.camera.bev_head import BEVDetectionHead
from utils.geometry import create_bev_grid


class BEVFormerLite(nn.Module):
    def __init__(
        self,
        bev_bounds: Dict[str, Tuple[float, float]],
        bev_resolution: float,
        num_cams: int,
        backbone_name: str,
        embed_dim: int,
        num_classes: int,
        bev_encoder_layers: int = 4,
        bev_num_heads: int = 8,
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (224, 224),
        attn_chunk_size: int = 256,
        max_attn_elements: int = 25_000_000,
    ) -> None:
        super().__init__()
        self.num_cams = num_cams
        self.embed_dim = embed_dim
        self.attn_chunk_size = attn_chunk_size
        self.max_attn_elements = max_attn_elements
        self.backbone = SwinBackbone(
            model_name=backbone_name,
            embed_dim=embed_dim,
            image_size=image_size,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=bev_num_heads, dropout=dropout, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=bev_num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.bev_encoder = nn.TransformerEncoder(encoder_layer, num_layers=bev_encoder_layers)
        self.head = BEVDetectionHead(embed_dim=embed_dim, num_classes=num_classes)

        x_bounds = bev_bounds["x"]
        y_bounds = bev_bounds["y"]
        bev_w = int((x_bounds[1] - x_bounds[0]) / bev_resolution)
        bev_h = int((y_bounds[1] - y_bounds[0]) / bev_resolution)
        self.bev_shape = (bev_h, bev_w)
        self.register_buffer(
            "bev_queries",
            torch.zeros(bev_h * bev_w, embed_dim),
        )
        grid = create_bev_grid(x_bounds, y_bounds, bev_resolution, device=torch.device("cpu"))
        self.register_buffer("bev_coords", grid, persistent=False)

        self.pos_embed = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, num_cams, _, _, _ = images.shape
        images = images.view(bsz * num_cams, *images.shape[2:])
        features = self.backbone(images)[-1]  # use highest-level feature
        _, c, h, w = features.shape
        features = features.view(bsz, num_cams, c, h, w)
        tokens = []
        for cam in range(num_cams):
            feat = features[:, cam]
            if self.camera_token_stride > 1:
                feat = F.avg_pool2d(
                    feat, kernel_size=self.camera_token_stride, stride=self.camera_token_stride
                )
            token = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
            tokens.append(token)
        camera_tokens = torch.cat(tokens, dim=1)

        bev_queries = self.bev_queries.unsqueeze(0).expand(bsz, -1, -1)
        query = bev_queries + self.pos_embed(bev_queries)
        bev_with_image = self._cross_attend(query, camera_tokens)
        bev_latent = self.bev_encoder(bev_with_image)
        cls_logits, box_preds = self.head(bev_latent)

        return {
            "bev_features": bev_latent,
            "cls_logits": cls_logits.view(bsz, *self.bev_shape, -1),
            "box_preds": box_preds.view(bsz, *self.bev_shape, -1),
        }

    def _cross_attend(self, query: torch.Tensor, camera_tokens: torch.Tensor) -> torch.Tensor:
        """Run cross attention in manageable chunks to limit memory overhead."""
        chunk = self.attn_chunk_size or query.shape[1]
        max_chunk = max(
            1,
            int(
                self.max_attn_elements
                // max(1, query.shape[0] * self.cross_attn.num_heads * camera_tokens.shape[1])
            ),
        )

        if chunk <= 0 or chunk >= query.shape[1]:
            chunk = query.shape[1]

        if chunk > max_chunk:
            if not hasattr(self, "_warned_chunk"):
                print(
                    f"[BEVFormerLite] Reducing cross-attn chunk size from {chunk} to {max_chunk} "
                    "to avoid excessive attention memory."
                )
                self._warned_chunk = True
            chunk = max_chunk

        if chunk >= query.shape[1]:
            attn_output, _ = self.cross_attn(
                query, camera_tokens, camera_tokens, need_weights=False
            )
            return attn_output

        outputs = []
        # Project K/V once to avoid redundant allocations for each chunk
        k_proj = F.linear(
            camera_tokens,
            self.cross_attn.in_proj_weight[self.embed_dim : 2 * self.embed_dim],
            self.cross_attn.in_proj_bias[self.embed_dim : 2 * self.embed_dim]
            if self.cross_attn.in_proj_bias is not None
            else None,
        )
        v_proj = F.linear(
            camera_tokens,
            self.cross_attn.in_proj_weight[2 * self.embed_dim :],
            self.cross_attn.in_proj_bias[2 * self.embed_dim :]
            if self.cross_attn.in_proj_bias is not None
            else None,
        )
        k_proj = k_proj.view(query.shape[0], -1, self.cross_attn.num_heads, self.cross_attn.head_dim)
        v_proj = v_proj.view(query.shape[0], -1, self.cross_attn.num_heads, self.cross_attn.head_dim)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        for start in range(0, query.shape[1], chunk):
            end = min(start + chunk, query.shape[1])
            q_proj = F.linear(
                query[:, start:end, :],
                self.cross_attn.in_proj_weight[: self.embed_dim],
                self.cross_attn.in_proj_bias[: self.embed_dim]
                if self.cross_attn.in_proj_bias is not None
                else None,
            )
            q_proj = q_proj.view(query.shape[0], -1, self.cross_attn.num_heads, self.cross_attn.head_dim)
            q_proj = q_proj.transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                q_proj,
                k_proj,
                v_proj,
                dropout_p=self.cross_attn.dropout if self.training else 0.0,
            )
            attn_output = attn_output.transpose(1, 2).reshape(query.shape[0], -1, self.embed_dim)
            attn_output = self.cross_attn.out_proj(attn_output)
            outputs.append(attn_output)
        return torch.cat(outputs, dim=1)
