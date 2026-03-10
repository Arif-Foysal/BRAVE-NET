"""
baselines.py
============
Baseline models for the BRAVE-Net ablation study.

| Configuration | Signal Input   | Features        | Classifier   | Role                       |
|---------------|----------------|-----------------|--------------|----------------------------|
| Baseline 1    | Raw Audio      | MFCCs (13 coef) | Random Forest| Replicates Hassan et al.   |
| Baseline 2    | Raw Audio      | Mel-Spectrogram | ResNet-18    | Standard deep learning     |
| Baseline 3    | Raw Audio      | Mel-Spectrogram | ViT-B/16     | Isolates transformer gain  |
| Proposed      | Burg-Restored  | Mel-Spectrogram | ViT-B/16     | Full BRAVE-Net             |

Baselines 3 and Proposed share the same ViT architecture;
the only difference is the input preprocessing pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from timm import create_model


# ─── Baseline 1: Random Forest + MFCCs ───────────────────────────────────────

def build_rf_classifier(
    n_estimators: int = 500,
    max_depth: int | None = None,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """Random Forest classifier for MFCC features.

    Replicates the approach of Hassan et al. (2019).

    Parameters
    ----------
    n_estimators : int
    max_depth : int or None
    random_state : int
    class_weight : str   'balanced' handles PD/HC imbalance.

    Returns
    -------
    clf : sklearn RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )


# ─── Baseline 2: ResNet-18 ────────────────────────────────────────────────────

class ResNet18Baseline(nn.Module):
    """ResNet-18 classifier on raw Mel-Spectrograms.

    Parameters
    ----------
    num_classes : int    Default 2 (PD / HC).
    pretrained : bool    Use torchvision ImageNet weights.
    dropout_rate : float Dropout before final layer.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Replace the final FC layer
        in_features = backbone.fc.in_features    # 512
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─── Baseline 3: ViT-B/16 on Raw Audio (no Burg restoration) ─────────────────

class ViTBaseline(nn.Module):
    """ViT-B/16 classifier without Burg signal restoration.

    Identical architecture to BRAVE-Net, but the input Mel-Spectrograms
    are derived from *raw* (unrestored) audio.  Comparing this baseline
    against BRAVE-Net isolates the contribution of the Burg restoration step.

    Parameters
    ----------
    backbone : str
    pretrained : bool
    num_classes : int
    unfreeze_last_n_blocks : int
    dropout_rate : float
    head_hidden_dim : int
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 2,
        unfreeze_last_n_blocks: int = 2,
        dropout_rate: float = 0.3,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = create_model(backbone, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.embed_dim  # 768

        # Freeze everything then unfreeze last N blocks
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in list(self.backbone.blocks)[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_baseline(name: str, config: dict) -> nn.Module | RandomForestClassifier:
    """Build a baseline model by name.

    Parameters
    ----------
    name : str
        One of: 'rf', 'resnet18', 'vit_raw'
    config : dict

    Returns
    -------
    model
    """
    model_cfg = config.get("model", {})
    if name == "rf":
        return build_rf_classifier()
    elif name == "resnet18":
        return ResNet18Baseline(
            num_classes=model_cfg.get("num_classes", 2),
            pretrained=model_cfg.get("pretrained", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
        )
    elif name == "vit_raw":
        return ViTBaseline(
            backbone=model_cfg.get("backbone", "vit_base_patch16_224"),
            pretrained=model_cfg.get("pretrained", True),
            num_classes=model_cfg.get("num_classes", 2),
            unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 2),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            head_hidden_dim=model_cfg.get("head_hidden_dim", 256),
        )
    else:
        raise ValueError(f"Unknown baseline name: '{name}'. Choose from: rf, resnet18, vit_raw")
