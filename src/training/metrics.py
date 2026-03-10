"""
metrics.py
==========
Clinical evaluation metrics for PD detection.

Metrics reported (per paper requirements):
  1. Sensitivity (Recall for PD class)   — primary clinical metric
  2. Specificity (Recall for HC class)   — primary clinical metric
  3. Accuracy
  4. F1-Score                            — primary comparative metric
  5. AUC-ROC
  6. Matthew's Correlation Coefficient (MCC)   — robust to class imbalance
  7. 95% Confidence Intervals via bootstrap (n=1000)

All metrics follow leave-one-speaker-out (LOSO) protocol.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def compute_sensitivity_specificity(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
) -> tuple[float, float]:
    """Compute sensitivity and specificity from binary labels.

    Sensitivity = TP / (TP + FN)   (recall for PD class, label=1)
    Specificity = TN / (TN + FP)   (recall for HC class, label=0)

    Parameters
    ----------
    y_true : ndarray, shape (N,)   Ground-truth labels (0=HC, 1=PD).
    y_pred : ndarray, shape (N,)   Predicted labels.

    Returns
    -------
    sensitivity : float
    specificity : float
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm[i, j] = number of samples with true label i predicted as j
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    return float(sensitivity), float(specificity)


def compute_all_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_prob: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Compute the full set of evaluation metrics.

    Parameters
    ----------
    y_true : ndarray, shape (N,)       Ground-truth binary labels.
    y_pred : ndarray, shape (N,)       Predicted binary labels.
    y_prob : ndarray, shape (N,) | None   Predicted probabilities for PD class.
                                          Required for AUC-ROC.

    Returns
    -------
    metrics : dict with keys:
        accuracy, sensitivity, specificity, f1, auc_roc, mcc
    """
    sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred)

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


# ─── Bootstrap Confidence Intervals ──────────────────────────────────────────

def bootstrap_ci(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_prob: NDArray[np.float64] | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute 95% confidence intervals for all metrics via bootstrapping.

    Parameters
    ----------
    y_true, y_pred, y_prob : arrays (see ``compute_all_metrics``).
    n_bootstrap : int   Number of bootstrap resamples.
    ci : float          Confidence level (default 0.95 → 95% CI).
    seed : int

    Returns
    -------
    ci_results : dict mapping metric_name → {'mean', 'lower', 'upper'}
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)
    alpha = (1.0 - ci) / 2.0

    # Collect bootstrapped metric values
    boot_metrics: dict[str, list[float]] = {
        k: [] for k in ["accuracy", "sensitivity", "specificity", "f1", "auc_roc", "mcc"]
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        bt_true = y_true[idx]
        bt_pred = y_pred[idx]
        bt_prob = y_prob[idx] if y_prob is not None else None

        # Skip degenerate resamples (only one class)
        if len(np.unique(bt_true)) < 2:
            continue

        m = compute_all_metrics(bt_true, bt_pred, bt_prob)
        for k, v in m.items():
            if not np.isnan(v):
                boot_metrics[k].append(v)

    ci_results: dict[str, dict[str, float]] = {}
    for metric, values in boot_metrics.items():
        if len(values) == 0:
            ci_results[metric] = {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
        else:
            arr = np.array(values)
            ci_results[metric] = {
                "mean":  float(np.mean(arr)),
                "lower": float(np.quantile(arr, alpha)),
                "upper": float(np.quantile(arr, 1.0 - alpha)),
            }

    return ci_results


# ─── McNemar's Test ───────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: NDArray[np.int_],
    y_pred_a: NDArray[np.int_],
    y_pred_b: NDArray[np.int_],
) -> dict[str, float]:
    """McNemar's test for statistical significance of pairwise model comparison.

    Tests the null hypothesis that models A and B have the same error rate.

    Parameters
    ----------
    y_true   : Ground-truth labels.
    y_pred_a : Predictions from model A (e.g., BRAVE-Net).
    y_pred_b : Predictions from model B (e.g., Baseline 3).

    Returns
    -------
    result : dict with 'statistic', 'p_value', 'significant' (at α=0.05)
    """
    from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

    # Contingency table
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # n01 = A correct, B wrong; n10 = A wrong, B correct
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))

    table = [[np.sum(correct_a & correct_b), n01],
             [n10, np.sum(~correct_a & ~correct_b)]]

    result = _mcnemar(table, exact=False, correction=True)
    return {
        "statistic":   float(result.statistic),
        "p_value":     float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


# ─── LOSO Aggregation ─────────────────────────────────────────────────────────

def aggregate_loso_results(
    per_fold_results: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-fold LOSO metrics into mean ± std.

    Parameters
    ----------
    per_fold_results : list of metric dicts, one per LOSO fold.

    Returns
    -------
    summary : dict mapping metric → {'mean', 'std', 'min', 'max'}
    """
    all_keys = per_fold_results[0].keys()
    summary: dict[str, dict[str, float]] = {}

    for key in all_keys:
        # Skip non-scalar entries (e.g. y_true, y_pred, y_prob lists)
        sample_val = per_fold_results[0].get(key)
        if isinstance(sample_val, (list, np.ndarray)):
            continue
        vals = np.array([r[key] for r in per_fold_results if not np.isnan(r.get(key, float("nan")))])
        if len(vals) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        else:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }

    return summary
