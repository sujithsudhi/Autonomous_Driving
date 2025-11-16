from typing import List, Sequence

import torch
import torch.nn as nn


class SwinBackbone(nn.Module):
    """Wrapper around timm Swin / ViT backbones returning multi-scale features."""

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        embed_dim: int = 256,
        image_size: Sequence[int] = (224, 224),
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for SwinBackbone. Please install `timm`.") from exc
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=image_size,
            dynamic_img_size=True,
        )
        self.projections = nn.ModuleList(
            [nn.LazyConv2d(embed_dim, kernel_size=1) for _ in range(len(out_indices))]
        )

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        feats = self.model(images)
        return [proj(feat) for proj, feat in zip(self.projections, feats)]
