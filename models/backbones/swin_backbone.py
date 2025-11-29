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
            features_only=False,
            img_size=image_size,
            num_classes=0,
            dynamic_img_size=True,
        )
        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(0)
        num_feats = getattr(self.model, "num_features", embed_dim)
        self.projection = nn.Conv2d(num_feats, embed_dim, kernel_size=1)
        first_block = self.model.layers[0].blocks[0]
        self._base_window_size = getattr(first_block, "window_size", (7, 7))

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        feats = self.model.forward_features(images)
        if feats.dim() == 4 and feats.shape[1] != self.projection.in_channels:
            # timm Swin returns NHWC, convert to NCHW
            feats = feats.permute(0, 3, 1, 2).contiguous()
        feats = self.projection(feats)
        return [feats]

    def set_image_size(self, image_size: Sequence[int]) -> None:
        """Update Swin's internal resolution metadata."""
        if hasattr(self.model, "set_input_size"):
            self.model.set_input_size(img_size=tuple(image_size), window_size=self._base_window_size)
        elif hasattr(self.model, "patch_embed"):
            self.model.patch_embed.set_input_size(img_size=tuple(image_size))
