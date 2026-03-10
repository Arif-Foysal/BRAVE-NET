"""
brave_net.py
============
BRAVE-Net: Burg Residual Augmented Vision Transformer.

Architecture
------------
Backbone : ViT-B/16 pre-trained on ImageNet-21k (via timm).

Freezing strategy (to prevent overfitting on small clinical datasets):
  1. All multi-head self-attention blocks are frozen by default.
  2. The last ``unfreeze_last_n_blocks`` transformer encoder blocks are
     unfrozen and fine-tuned (gradual unfreezing).
  3. Only the custom classification head is always trainable.

Classification head:
  CLS token → Linear(768, 256) → GELU → Dropout → Linear(256, num_classes)

Input : 3-channel Mel-Spectrogram image, shape (B, 3, 224, 224)
Output: logits, shape (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm import create_model
from timm.models.vision_transformer import VisionTransformer


class BraveNet(nn.Module):
    """BRAVE-Net classifier.

    Parameters
    ----------
    backbone : str
        timm model name. Default: 'vit_base_patch16_224'.
    pretrained : bool
        Load ImageNet-21k pre-trained weights.
    num_classes : int
        Number of output classes (2 for PD/HC).
    freeze_attention_blocks : bool
        If True, freeze all attention blocks in the backbone.
    unfreeze_last_n_blocks : int
        Number of transformer encoder blocks to unfreeze from the end.
    dropout_rate : float
        Dropout before the final classification layer.
    head_hidden_dim : int
        Hidden dimension of the classification head MLP.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_attention_blocks: bool = True,
        unfreeze_last_n_blocks: int = 2,
        dropout_rate: float = 0.3,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # ── Load backbone ─────────────────────────────────────────────────────
        # num_classes=0 removes timm's default head; we add our own
        self.backbone: VisionTransformer = create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # removes default head
        )
        embed_dim = self.backbone.embed_dim  # 768 for ViT-B/16

        # ── Apply freezing strategy ───────────────────────────────────────────
        if freeze_attention_blocks:
            self._freeze_backbone(unfreeze_last_n_blocks)

        # ── Custom classification head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_classes),
        )

        # Initialise the custom head with Xavier uniform
        self._init_head()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 3, 224, 224)
            Normalised Mel-Spectrogram images.

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
        """
        # ViT backbone returns the CLS token representation
        features = self.backbone(x)        # (B, embed_dim)
        logits   = self.classifier(features)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features (before the head) for analysis."""
        return self.backbone(x)

    # ── Freezing ──────────────────────────────────────────────────────────────

    def _freeze_backbone(self, unfreeze_last_n: int) -> None:
        """Freeze all backbone parameters, then selectively unfreeze last N blocks."""
        # Step 1: Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze the last N transformer encoder blocks
        blocks = list(self.backbone.blocks)  # list of Block modules
        for block in blocks[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Step 3: Always unfreeze the norm layer and positional embedding
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the entire backbone (for full fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_all_except_head(self) -> None:
        """Freeze everything except the classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_head(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Info ──────────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"BraveNet(\n"
            f"  backbone={self.backbone.__class__.__name__},\n"
            f"  total_params={counts['total']:,},\n"
            f"  trainable_params={counts['trainable']:,}\n"
            f")"
        )


def build_model(config: dict) -> BraveNet:
    """Instantiate BraveNet from a config dict.

    Parameters
    ----------
    config : dict   Loaded YAML config (see configs/brave_net_config.yaml).

    Returns
    -------
    model : BraveNet
    """
    model_cfg = config["model"]
    return BraveNet(
        backbone=model_cfg.get("backbone", "vit_base_patch16_224"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 2),
        freeze_attention_blocks=model_cfg.get("freeze_attention_blocks", True),
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 2),
        dropout_rate=model_cfg.get("dropout_rate", 0.3),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 256),
    )
