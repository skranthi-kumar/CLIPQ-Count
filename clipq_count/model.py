"""CLIPQ-Count model: frozen CLIP ViT-B/32 (last two vision blocks trainable) + query-guided fusion heads."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import CLIPModel

from .data import CLIP_NAME, IMAGE_SIZE
from .modules import PointGuidedAttention, SelfAdaptiveFeatureEnhancement

# Hugging Face CLIP parameter names for the last two vision-encoder blocks.
# NOTE: the original research script used OpenAI-CLIP names ("visual.transformer.resblocks.N"),
# which never match HF parameters, so CLIP stayed fully frozen there. Pass trainable_blocks=()
# to reproduce that behaviour exactly.
TRAINABLE_BLOCKS = ("vision_model.encoder.layers.10", "vision_model.encoder.layers.11")


class CLIPCountingModel(nn.Module):
    def __init__(self, shots: int = 3, dropout: float = 0.2, clip_name: str = CLIP_NAME, trainable_blocks=TRAINABLE_BLOCKS):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.text_dim = self.clip.config.projection_dim          # 512 for ViT-B/32
        self.image_dim = self.clip.config.vision_config.hidden_size  # 768 for ViT-B/32
        self.shots = shots

        for name, param in self.clip.named_parameters():
            param.requires_grad = any(block in name for block in trainable_blocks)

        self.feature_norm = nn.LayerNorm(self.text_dim)
        self.point_attention = PointGuidedAttention(self.image_dim)
        self.sfe = SelfAdaptiveFeatureEnhancement(self.image_dim, self.text_dim)

        d = self.text_dim
        self.fusion = nn.Sequential(
            nn.Linear(d * 3, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.count_head = nn.Sequential(
            nn.Linear(d // 2, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.density_head = nn.Sequential(
            nn.Conv2d(self.image_dim, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid(),
        )
        self._init_heads()

    def _init_heads(self):
        for m in (self.fusion, self.count_head, self.density_head, self.point_attention, self.sfe):
            for layer in m.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _patch_grid(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP vision tokens (minus CLS) reshaped to a B x C x S x S grid."""
        tokens = self.clip.vision_model(pixel_values=images).last_hidden_state
        b, n, c = tokens.shape
        s = int(math.sqrt(n - 1))
        return tokens[:, 1:, :].permute(0, 2, 1).reshape(b, c, s, s)

    def forward(self, images, exemplars, text, points):
        device_type = "cuda" if images.is_cuda else "cpu"
        with autocast(device_type=device_type):
            b = images.size(0)

            grid = self.point_attention(self._patch_grid(images), points)
            attended = grid.mean(dim=(2, 3))

            text_feat = self.feature_norm(self.clip.get_text_features(**text))

            _, k, c, h, w = exemplars.shape
            ex_feat = self.clip.get_image_features(pixel_values=exemplars.view(b * k, c, h, w))
            ex_feat = self.feature_norm(ex_feat).view(b, k, -1).mean(dim=1)

            fused = self.fusion(torch.cat([self.sfe(attended, text_feat), text_feat, ex_feat], dim=-1))
            log_count = self.count_head(fused)
            count = torch.clamp(torch.exp(log_count.squeeze(-1)) - 1, min=0)

            density = F.interpolate(self.density_head(grid), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            return count, log_count, density
