"""
visualize.py
============
Visualisation utilities for BRAVE-Net.

Includes:
  1. Grad-CAM heatmaps on Mel-Spectrogram images (ViT-compatible)
  2. Confusion matrix plots
  3. ROC curve plots
  4. Signal restoration comparison (raw vs Burg-restored waveform / spectrogram)
  5. LOSO results bar charts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_curve, confusion_matrix


# ─── Grad-CAM for ViT ────────────────────────────────────────────────────────

def get_vit_gradcam_target_layers(model: nn.Module) -> list:
    """Return the target layer(s) for Grad-CAM on a ViT backbone.

    For ViT-B/16, the last transformer block's LayerNorm is the standard
    choice for generating spatially meaningful activation maps.

    Parameters
    ----------
    model : BraveNet or ViTBaseline

    Returns
    -------
    target_layers : list of nn.Module
    """
    backbone = getattr(model, "backbone", model)
    # Last transformer block
    last_block = list(backbone.blocks)[-1]
    return [last_block.norm1]


def reshape_transform_vit(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Reshape ViT token sequence to spatial feature map for Grad-CAM.

    ViT outputs [CLS token + patch tokens] → we strip the CLS and reshape.

    Parameters
    ----------
    tensor : Tensor, shape (B, num_patches+1, embed_dim)
    height, width : int   Grid size (14×14 for ViT-B/16 with 224×224 input).

    Returns
    -------
    Tensor, shape (B, embed_dim, height, width)
    """
    result = tensor[:, 1:, :]  # Drop CLS token → (B, 196, 768)
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)  # → (B, 768, 14, 14)
    return result


def generate_gradcam_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: str = "cpu",
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image.

    Uses the ``pytorch-grad-cam`` library with ViT-compatible reshape transform.

    Parameters
    ----------
    model : BraveNet or ViTBaseline
    image_tensor : Tensor, shape (1, 3, 224, 224)
    target_class : int   Class index to generate attribution for (1 = PD).
    device : str

    Returns
    -------
    heatmap : ndarray, shape (224, 224)   Values in [0, 1].
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layers = get_vit_gradcam_target_layers(model)
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform_vit,
    )
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(
        input_tensor=image_tensor.to(device),
        targets=targets,
    )
    return grayscale_cam[0]  # shape (224, 224)


def plot_gradcam_overlay(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    title: str = "Grad-CAM",
    save_path: str | Path | None = None,
    y_true: int | None = None,
    y_pred: int | None = None,
) -> plt.Figure:
    """Plot Grad-CAM heatmap overlaid on the Mel-Spectrogram image.

    Parameters
    ----------
    image_tensor : Tensor, shape (1, 3, 224, 224) or (3, 224, 224)
    heatmap : ndarray, shape (224, 224)
    title : str
    save_path : Path or None
    y_true, y_pred : int (0=HC, 1=PD) for annotation

    Returns
    -------
    fig : matplotlib Figure
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # De-normalise image (reverse ImageNet normalisation)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std + mean).clip(0, 1).astype(np.float32)

    cam_image = show_cam_on_image(img_np, heatmap, use_rgb=True)

    label_names = {0: "HC", 1: "PD"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Mel-Spectrogram")
    axes[0].axis("off")

    axes[1].imshow(cam_image)
    subtitle = title
    if y_true is not None and y_pred is not None:
        colour = "green" if y_true == y_pred else "red"
        subtitle += f"\nTrue: {label_names[y_true]} | Pred: {label_names[y_pred]}"
        axes[1].set_title(subtitle, color=colour)
    else:
        axes[1].set_title(subtitle)
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix.

    Parameters
    ----------
    y_true, y_pred : arrays of int labels.
    class_names : list of str  (default ['HC', 'PD'])
    title : str
    save_path : Path or None

    Returns
    -------
    fig : Figure
    """
    if class_names is None:
        class_names = ["HC", "PD"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── ROC Curve ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    models_results: dict[str, dict],
    title: str = "ROC Curves — Ablation Study",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    models_results : dict mapping model_name → eval_results dict
        Each value must contain 'y_true', 'y_prob', 'metrics'.
    title : str
    save_path : Path or None

    Returns
    -------
    fig : Figure
    """
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, res) in enumerate(models_results.items()):
        y_true = np.array(res["y_true"])
        y_prob = np.array(res["y_prob"])
        if "metrics" in res:
            auc = res["metrics"].get("auc_roc", 0.0)
        else:
            from sklearn.metrics import roc_auc_score as _auc
            try:
                auc = _auc(y_true, y_prob)
            except ValueError:
                auc = 0.0

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Signal Restoration Visualisation ────────────────────────────────────────

def plot_restoration_comparison(
    raw_signal: np.ndarray,
    restored_signal: np.ndarray,
    sample_rate: int,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side comparison of raw vs Burg-restored signal.

    Shows: waveform, spectrogram, and LP residual for both signals.

    Parameters
    ----------
    raw_signal : ndarray, shape (N,)
    restored_signal : ndarray, shape (N,)
    sample_rate : int
    save_path : Path or None

    Returns
    -------
    fig : Figure
    """
    import librosa
    import librosa.display

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    time_axis = np.arange(len(raw_signal)) / sample_rate

    # --- Row 1: Waveforms ---
    axes[0, 0].plot(time_axis, raw_signal, color="#2196F3", lw=0.8, alpha=0.8)
    axes[0, 0].set_title("Raw Signal — Waveform", fontweight="bold")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(time_axis, restored_signal, color="#F44336", lw=0.8, alpha=0.8)
    axes[0, 1].set_title("Burg-Restored Signal — Waveform", fontweight="bold")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(alpha=0.3)

    # --- Row 2: Mel-Spectrograms ---
    def _mel_db(sig: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=sig.astype(np.float32), sr=sample_rate, n_mels=128
        )
        return librosa.power_to_db(mel, ref=np.max)

    raw_mel  = _mel_db(raw_signal)
    rest_mel = _mel_db(restored_signal)

    librosa.display.specshow(
        raw_mel, sr=sample_rate, hop_length=512,
        x_axis="time", y_axis="mel", ax=axes[1, 0], cmap="viridis"
    )
    axes[1, 0].set_title("Raw Signal — Mel-Spectrogram", fontweight="bold")

    librosa.display.specshow(
        rest_mel, sr=sample_rate, hop_length=512,
        x_axis="time", y_axis="mel", ax=axes[1, 1], cmap="viridis"
    )
    axes[1, 1].set_title("Burg-Restored Signal — Mel-Spectrogram", fontweight="bold")

    plt.suptitle(
        "Signal Restoration: PD Biomarker Preservation via Burg Residual Enhancement",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── LOSO Results Bar Chart ───────────────────────────────────────────────────

def plot_loso_metric_comparison(
    models_aggregate: dict[str, dict] = None,
    metric: str = "f1",
    save_path: str | Path | None = None,
    *,
    results_dict: dict[str, dict] | None = None,
) -> plt.Figure:
    """Bar chart comparing LOSO mean ± std for a given metric across models.

    Parameters
    ----------
    models_aggregate : dict mapping model_name → aggregate metrics dict
        (output of ``aggregate_loso_results``)
    metric : str   e.g. 'f1', 'sensitivity', 'auc_roc'
    save_path : Path or None

    Returns
    -------
    fig : Figure
    """
    # Support both positional 'models_aggregate' and keyword 'results_dict'
    data = models_aggregate if models_aggregate is not None else results_dict
    if data is None:
        raise ValueError("Must provide models_aggregate or results_dict")
    names  = list(data.keys())
    # Support both nested {metric: {"mean": ..., "std": ...}} and flat {metric: float}
    means, stds = [], []
    for n in names:
        v = data[n].get(metric, 0)
        if isinstance(v, dict):
            means.append(v.get("mean", 0))
            stds.append(v.get("std", 0))
        else:
            means.append(float(v))
            stds.append(0.0)

    colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#EF5350"]  # Highlight BRAVE-Net

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, edgecolor="black", lw=0.8)

    ax.set_ylabel(metric.upper().replace("_", " "), fontsize=12)
    ax.set_title(f"LOSO-CV {metric.upper()} Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(color="#90CAF9", label="Baselines"),
        mpatches.Patch(color="#EF5350", label="BRAVE-Net (Proposed)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
