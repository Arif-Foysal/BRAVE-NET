"""
losses.py
=========
Custom loss functions for BRAVE-Net training.

The TORGO dataset is inherently imbalanced (more control utterances than
dysarthric utterances), so weighted cross-entropy is used as the primary
training objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
import numpy as np


def compute_class_weights(
    labels: list[int] | NDArray[np.int_],
    num_classes: int = 2,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted cross-entropy.

    w_c = N / (num_classes * count_c)

    Parameters
    ----------
    labels : array-like of int labels.
    num_classes : int

    Returns
    -------
    weights : FloatTensor, shape (num_classes,)
    """
    labels_arr = np.asarray(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    N = len(labels_arr)
    for c in range(num_classes):
        count = np.sum(labels_arr == c)
        weights[c] = N / (num_classes * count) if count > 0 else 1.0
    return torch.tensor(weights, dtype=torch.float32)


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss to handle PD/HC class imbalance.

    Parameters
    ----------
    class_weights : FloatTensor or None
        Per-class weights.  If None, standard (unweighted) CE is used.
    label_smoothing : float
        Label smoothing factor in [0, 1).  Default 0.1.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor, shape (B, num_classes)
        targets : LongTensor, shape (B,)

        Returns
        -------
        loss : scalar Tensor
        """
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device)
        return F.cross_entropy(
            logits,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )


def build_loss(
    labels: list[int] | NDArray[np.int_],
    label_smoothing: float = 0.1,
    device: str = "cpu",
) -> WeightedCrossEntropyLoss:
    """Build a weighted CE loss from training labels.

    Parameters
    ----------
    labels : training split labels (for computing class weights).
    label_smoothing : float
    device : str

    Returns
    -------
    criterion : WeightedCrossEntropyLoss
    """
    weights = compute_class_weights(labels).to(device)
    return WeightedCrossEntropyLoss(class_weights=weights, label_smoothing=label_smoothing)
