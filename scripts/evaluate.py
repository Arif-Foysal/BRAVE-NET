"""
evaluate.py (script)
====================
Post-training evaluation script.

Loads a saved checkpoint, runs inference on the test set, generates
Grad-CAM heatmaps, and produces all publication-quality figures.

Usage
-----
    python scripts/evaluate.py \
        --checkpoint checkpoints/brave_net_loso_M01_best.pt \
        --manifest   data/processed/TORGO/brave_net/test_M01_manifest.json \
        --model      brave_net \
        --config     configs/brave_net_config.yaml \
        --output_dir results/brave_net_eval \
        --gradcam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BRAVE-Net and generate figures")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest",   required=True, help="Path to test manifest JSON")
    parser.add_argument("--model",      choices=["brave_net", "resnet18", "vit_raw"],
                        default="brave_net")
    parser.add_argument("--config",     default="configs/brave_net_config.yaml")
    parser.add_argument("--output_dir", default="results/evaluation")
    parser.add_argument("--gradcam",    action="store_true",
                        help="Generate Grad-CAM heatmaps for misclassified samples")
    parser.add_argument("--n_gradcam",  type=int, default=10,
                        help="Number of Grad-CAM images to generate")
    parser.add_argument("--device",     default=None)
    return parser.parse_args()


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args   = parse_args()
    device = get_device(args.device)

    from src.utils.config import load_config
    from src.utils.dataset import BraveNetDataset, get_vit_transforms
    from src.evaluation.evaluate import evaluate_model
    from src.evaluation.visualize import (
        plot_confusion_matrix,
        plot_roc_curves,
        plot_gradcam_overlay,
        generate_gradcam_heatmap,
    )

    config    = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    if args.model == "brave_net":
        from src.models.brave_net import build_model
        model = build_model(config)
    elif args.model == "resnet18":
        from src.models.baselines import ResNet18Baseline
        model = ResNet18Baseline(num_classes=config["model"]["num_classes"])
    elif args.model == "vit_raw":
        from src.models.baselines import ViTBaseline
        model = ViTBaseline(
            backbone=config["model"]["backbone"],
            num_classes=config["model"]["num_classes"],
        )

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"\n[Loaded] {args.checkpoint}")

    # ── Load test dataset ─────────────────────────────────────────────────────
    test_ds = BraveNetDataset(args.manifest, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=(device == "cuda"),
    )
    print(f"[Test set] {len(test_ds)} samples")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = evaluate_model(model, test_loader, device=device)
    metrics = results["metrics"]
    ci      = results["ci"]

    print(f"\n{'═'*55}")
    print(f"  Evaluation Results — {args.model}")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        lo = ci.get(k, {}).get("lower", 0)
        hi = ci.get(k, {}).get("upper", 0)
        print(f"  {k:<14}: {v:.4f}  [95% CI: {lo:.4f} – {hi:.4f}]")
    print(f"{'═'*55}\n")

    # Save results
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "ci": ci}, f, indent=2)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    y_true = np.array(results["y_true"])
    y_pred = np.array(results["y_pred"])
    y_prob = np.array(results["y_prob"])

    fig_cm = plot_confusion_matrix(
        y_true, y_pred,
        title=f"Confusion Matrix — {args.model}",
        save_path=output_dir / "confusion_matrix.png",
    )
    print(f"[Saved] confusion_matrix.png")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fig_roc = plot_roc_curves(
        {args.model: results},
        title=f"ROC Curve — {args.model}",
        save_path=output_dir / "roc_curve.png",
    )
    print(f"[Saved] roc_curve.png")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if args.gradcam and args.model in ["brave_net", "vit_raw"]:
        gradcam_dir = output_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)
        print(f"\nGenerating Grad-CAM heatmaps ({args.n_gradcam} samples) ...")

        generated = 0
        for i in range(len(test_ds)):
            if generated >= args.n_gradcam:
                break
            img_tensor, label, meta = test_ds[i]
            img_4d = img_tensor.unsqueeze(0).to(device)
            heatmap = generate_gradcam_heatmap(model, img_4d, target_class=1, device=device)

            pred = int(y_pred[i])
            true = int(y_true[i])

            tag = "correct" if pred == true else "wrong"
            save_path = gradcam_dir / f"{i:04d}_{tag}_true{true}_pred{pred}.png"
            plot_gradcam_overlay(
                img_tensor, heatmap,
                title=f"Grad-CAM ({args.model})",
                save_path=save_path,
                y_true=true,
                y_pred=pred,
            )
            generated += 1

        print(f"[Saved] {generated} Grad-CAM images → {gradcam_dir}")

    print(f"\n[Done] All outputs saved → {output_dir}")


if __name__ == "__main__":
    main()
