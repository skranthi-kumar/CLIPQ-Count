"""Building blocks: point-guided spatial attention and query-conditioned feature gating."""
import torch
import torch.nn as nn


class PointGuidedAttention(nn.Module):
    """Gate CLIP patch features with a learned 1x1 attention map that is switched on at annotated points.

    During training the exemplar point annotations tell the model *where* instances are;
    the conv learns what those locations look like so the gate generalises at test time.
    """

    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, points) -> torch.Tensor:
        b, _, h, w = features.shape
        mask = torch.zeros(b, 1, h, w, device=features.device)
        for i, pts in enumerate(points):
            if pts.shape[0] == 0:
                continue
            pts = pts.to(features.device) * torch.tensor([h, w], device=features.device)
            for p in pts:
                x, y = p.long()
                if 0 <= x < h and 0 <= y < w:
                    mask[i, 0, x, y] = 1.0
        attention = self.sigmoid(self.conv(features) * mask)
        return features * attention


class SelfAdaptiveFeatureEnhancement(nn.Module):
    """Project image features to the text space and gate them by the text query (sigmoid gate)."""

    def __init__(self, image_dim: int = 768, text_dim: int = 512):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, text_dim)
        self.gate = nn.Sequential(nn.Linear(text_dim * 2, text_dim), nn.Sigmoid())

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        projected = self.image_proj(image_features)
        gate = self.gate(torch.cat([projected, text_features], dim=-1))
        return projected * gate
