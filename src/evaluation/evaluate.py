"""
evaluate.py
===========
Evaluation pipeline for BRAVE-Net.

Runs inference on a test set (or full LOSO split), collects predictions,
computes all metrics with 95% confidence intervals, and saves results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..training.metrics import compute_all_metrics, bootstrap_ci, aggregate_loso_results


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    return_embeddings: bool = False,
) -> dict[str, Any]:
    """Run inference on a DataLoader and return predictions + metrics.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : str
    return_embeddings : bool
        If True and model has a ``get_features`` method, also return CLS tokens.

    Returns
    -------
    results : dict with keys:
        'metrics', 'y_true', 'y_pred', 'y_prob', 'embeddings' (optional),
        'ci' (bootstrap 95% CI for all metrics)
    """
    model.eval()
    model.to(device)

    all_labels, all_preds, all_probs = [], [], []
    all_embeddings = [] if return_embeddings else None

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            if return_embeddings and hasattr(model, "get_features"):
                emb = model.get_features(images).cpu().numpy()
                all_embeddings.append(emb)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    ci      = bootstrap_ci(y_true, y_pred, y_prob)

    result: dict[str, Any] = {
        "metrics": metrics,
        "ci":      ci,
        "y_true":  y_true.tolist(),
        "y_pred":  y_pred.tolist(),
        "y_prob":  y_prob.tolist(),
    }
    if return_embeddings and all_embeddings:
        result["embeddings"] = np.concatenate(all_embeddings, axis=0).tolist()

    return result


def run_ablation_evaluation(
    results_dir: str | Path,
    model_results: dict[str, dict],
) -> None:
    """Save and print the ablation table comparing all configurations.

    Parameters
    ----------
    results_dir : Path
    model_results : dict mapping config_name → LOSO aggregate metrics dict
        e.g., {'baseline_rf': {...}, 'baseline_resnet18': {...},
               'baseline_vit_raw': {...}, 'brave_net': {...}}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*80}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'─'*80}")
    header = f"  {'Configuration':<22} | {'Acc':>6} | {'Sens':>6} | {'Spec':>6} | {'F1':>6} | {'AUC':>6} | {'MCC':>6}"
    print(header)
    print(f"{'─'*80}")

    for name, agg in model_results.items():
        def fmt(key: str) -> str:
            s = agg.get(key, {})
            if isinstance(s, dict):
                return f"{s.get('mean', 0):.4f}"
            return f"{s:.4f}"

        print(
            f"  {name:<22} | "
            f"{fmt('accuracy'):>6} | "
            f"{fmt('sensitivity'):>6} | "
            f"{fmt('specificity'):>6} | "
            f"{fmt('f1'):>6} | "
            f"{fmt('auc_roc'):>6} | "
            f"{fmt('mcc'):>6}"
        )

    print(f"{'═'*80}\n")

    save_path = results_dir / "ablation_table.json"
    with open(save_path, "w") as f:
        json.dump(model_results, f, indent=2)
    print(f"[Ablation table saved] → {save_path}")
