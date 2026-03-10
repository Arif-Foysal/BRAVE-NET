"""
train.py
========
Entry point for BRAVE-Net training experiments.

Usage Examples
--------------
# Train BRAVE-Net (proposed method):
    python scripts/train.py --model brave_net \
        --processed_dir data/processed/TORGO/brave_net \
        --config configs/brave_net_config.yaml \
        --run_name brave_net_exp1

# Train Baseline 2 (ResNet-18, raw Mel-Spectrograms):
    python scripts/train.py --model resnet18 \
        --processed_dir data/processed/TORGO/raw_mel \
        --config configs/brave_net_config.yaml \
        --run_name baseline_resnet18

# Train Baseline 3 (ViT, raw Mel-Spectrograms):
    python scripts/train.py --model vit_raw \
        --processed_dir data/processed/TORGO/raw_mel \
        --config configs/brave_net_config.yaml \
        --run_name baseline_vit_raw

# Train Baseline 1 (Random Forest + MFCCs):
    python scripts/train.py --model rf \
        --mfcc_path data/processed/TORGO/mfcc/features.npz \
        --config configs/brave_net_config.yaml \
        --run_name baseline_rf
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BRAVE-Net or baselines")
    parser.add_argument("--model",
                        choices=["brave_net", "resnet18", "vit_raw", "rf"],
                        default="brave_net")
    parser.add_argument("--processed_dir", default=None,
                        help="Directory with LOSO manifests (for deep models)")
    parser.add_argument("--mfcc_path", default=None,
                        help="Path to MFCC .npz file (for RF baseline only)")
    parser.add_argument("--config", default="configs/brave_net_config.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None,
                        help="'cuda', 'mps', 'cpu', or auto-detect if omitted")
    return parser.parse_args()


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_rf_baseline(mfcc_path: str, config: dict, run_name: str) -> None:
    """Train and evaluate Random Forest with LOSO-CV."""
    import json
    from sklearn.model_selection import LeaveOneGroupOut
    from src.models.baselines import build_rf_classifier
    from src.training.metrics import compute_all_metrics, aggregate_loso_results

    data = np.load(mfcc_path, allow_pickle=True)
    X       = data["features"]
    y       = data["labels"].astype(int)
    groups  = data["speakers"]

    loso    = LeaveOneGroupOut()
    per_fold = []

    for train_idx, test_idx in loso.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = build_rf_classifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = compute_all_metrics(y_test, y_pred, y_prob)
        per_fold.append(metrics)
        speaker = groups[test_idx[0]]
        print(f"  Fold {speaker}: F1={metrics['f1']:.4f} | "
              f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f}")

    aggregate = aggregate_loso_results(per_fold)
    print(f"\n  RF LOSO-CV F1: {aggregate['f1']['mean']:.4f} ± {aggregate['f1']['std']:.4f}")

    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_name}_loso_results.json"
    with open(out_path, "w") as f:
        json.dump({"per_fold": per_fold, "aggregate": aggregate}, f, indent=2)
    print(f"  [Saved] → {out_path}")


def main() -> None:
    args   = parse_args()
    device = get_device(args.device)

    from src.utils.config import load_config
    config = load_config(args.config)

    seed = config["training"]["seed"]
    set_seed(seed)

    run_name = args.run_name or args.model
    print(f"\n{'═'*60}")
    print(f"  BRAVE-Net Training")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print(f"  Run     : {run_name}")
    print(f"{'═'*60}\n")

    # ── Baseline 1: Random Forest ─────────────────────────────────────────────
    if args.model == "rf":
        if not args.mfcc_path:
            print("[ERROR] --mfcc_path required for rf baseline")
            sys.exit(1)
        train_rf_baseline(args.mfcc_path, config, run_name)
        return

    # ── Deep Learning Models ───────────────────────────────────────────────────
    if not args.processed_dir:
        print("[ERROR] --processed_dir required for deep learning models")
        sys.exit(1)

    from src.utils.dataset import scan_torgo_dataset, get_loso_splits
    from src.training.trainer import run_loso_cv

    # Scan for LOSO splits
    torgo_root = config["datasets"]["torgo"]["root"]
    records    = scan_torgo_dataset(torgo_root)
    splits     = get_loso_splits(records)

    # Model factory (called fresh per fold)
    def model_builder():
        if args.model == "brave_net":
            from src.models.brave_net import build_model
            return build_model(config)
        elif args.model == "resnet18":
            from src.models.baselines import ResNet18Baseline
            return ResNet18Baseline(
                num_classes=config["model"]["num_classes"],
                pretrained=config["model"]["pretrained"],
                dropout_rate=config["model"]["dropout_rate"],
            )
        elif args.model == "vit_raw":
            from src.models.baselines import ViTBaseline
            return ViTBaseline(
                backbone=config["model"]["backbone"],
                pretrained=config["model"]["pretrained"],
                num_classes=config["model"]["num_classes"],
                unfreeze_last_n_blocks=config["model"]["unfreeze_last_n_blocks"],
                dropout_rate=config["model"]["dropout_rate"],
                head_hidden_dim=config["model"]["head_hidden_dim"],
            )

    cv_results = run_loso_cv(
        model_builder=model_builder,
        splits=splits,
        processed_dir=args.processed_dir,
        config=config,
        device=device,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
