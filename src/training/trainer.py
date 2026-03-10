"""
trainer.py
==========
Training loop for BRAVE-Net with LOSO cross-validation.

Features
--------
  • Mixed-precision training (torch.cuda.amp)
  • Cosine annealing LR scheduler with linear warm-up
  • Early stopping (patience configurable in YAML)
  • Gradient clipping
  • Per-epoch logging: loss, accuracy, sensitivity, specificity, F1
  • Best-checkpoint saving (by validation F1)
  • Optional Weights & Biases integration
  • Leave-One-Speaker-Out cross-validation loop
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .losses import build_loss
from .metrics import compute_all_metrics, aggregate_loso_results


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full training + validation loop.

    Parameters
    ----------
    model : nn.Module
    config : dict   Full loaded YAML config.
    device : str    'cuda', 'mps', or 'cpu'.
    run_name : str  Tag for checkpoint/log naming.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: str = "cpu",
        run_name: str = "brave_net",
    ) -> None:
        self.model    = model.to(device)
        self.config   = config
        self.device   = device
        self.run_name = run_name

        train_cfg = config["training"]
        self.max_epochs        = train_cfg["max_epochs"]
        self.patience          = train_cfg["early_stopping_patience"]
        self.grad_clip         = train_cfg["gradient_clip_norm"]
        self.mixed_precision   = train_cfg["mixed_precision"] and (device == "cuda")
        self.log_interval      = config["logging"]["log_interval"]
        self.save_best_only    = config["logging"]["save_best_only"]

        # Paths
        self.checkpoint_dir = Path(config["paths"]["checkpoints"])
        self.log_dir        = Path(config["paths"]["logs"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # W&B (optional)
        self._wandb_enabled = config["logging"].get("use_wandb", False)
        if self._wandb_enabled:
            import wandb
            wandb.init(
                project=config["logging"]["wandb_project"],
                entity=config["logging"].get("wandb_entity", None),
                name=run_name,
                config=config,
            )

    # ── Optimiser + Scheduler Factory ────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        train_cfg = self.config["training"]
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        train_cfg = self.config["training"]
        warmup = train_cfg.get("warmup_epochs", 5) * steps_per_epoch
        total  = self.max_epochs * steps_per_epoch

        # Linear warm-up then cosine annealing
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── One Epoch ─────────────────────────────────────────────────────────────

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        train: bool,
    ) -> tuple[float, dict[str, float]]:
        """Run one train or validation epoch.

        Returns
        -------
        avg_loss : float
        metrics  : dict
        """
        self.model.train(train)
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        with torch.set_grad_enabled(train):
            for batch_idx, (images, labels, _) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    logits = self.model(images)
                    loss   = criterion(logits, labels)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    if scheduler is not None:
                        scheduler.step()

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())

        avg_loss = total_loss / len(loader)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        return avg_loss, metrics

    # ── Main Fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_labels: list[int],
        fold_tag: str = "fold",
    ) -> dict[str, Any]:
        """Train the model for one LOSO fold.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader   : DataLoader
        train_labels : list[int]   Training labels for class weight computation.
        fold_tag     : str         Tag for checkpoint naming (e.g., 'speaker_M01').

        Returns
        -------
        history : dict with 'best_val_metrics', 'epoch_logs'
        """
        criterion = build_loss(train_labels, device=self.device)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, len(train_loader))

        best_val_f1   = -1.0
        no_improve    = 0
        epoch_logs    = []
        best_metrics  = {}
        checkpoint_path = self.checkpoint_dir / f"{self.run_name}_{fold_tag}_best.pt"

        print(f"\n{'─'*60}")
        print(f"  Training fold: {fold_tag}")
        print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        print(f"{'─'*60}")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_loss, train_m = self._run_epoch(
                train_loader, criterion, optimizer, scheduler, train=True
            )
            val_loss, val_m = self._run_epoch(
                val_loader, criterion, optimizer=None, scheduler=None, train=False
            )

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            log = {
                "epoch":         epoch,
                "train_loss":    round(train_loss, 4),
                "val_loss":      round(val_loss, 4),
                "train_f1":      round(train_m["f1"], 4),
                "val_f1":        round(val_m["f1"], 4),
                "val_sens":      round(val_m["sensitivity"], 4),
                "val_spec":      round(val_m["specificity"], 4),
                "val_auc":       round(val_m["auc_roc"], 4),
                "lr":            round(lr, 8),
                "elapsed_s":     round(elapsed, 1),
            }
            epoch_logs.append(log)

            # Print progress
            print(
                f"  Epoch {epoch:>3}/{self.max_epochs} | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"F1 {train_m['f1']:.4f}/{val_m['f1']:.4f} | "
                f"Sens {val_m['sensitivity']:.4f} | "
                f"Spec {val_m['specificity']:.4f} | "
                f"AUC {val_m['auc_roc']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if self._wandb_enabled:
                import wandb
                wandb.log({f"{fold_tag}/{k}": v for k, v in log.items()})

            # Checkpoint
            if val_m["f1"] > best_val_f1:
                best_val_f1  = val_m["f1"]
                best_metrics = val_m
                no_improve   = 0
                if self.save_best_only:
                    torch.save(self.model.state_dict(), checkpoint_path)
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= self.patience:
                print(f"  [EarlyStopping] No improvement for {self.patience} epochs.")
                break

        return {
            "best_val_metrics": best_metrics,
            "epoch_logs":       epoch_logs,
            "checkpoint_path":  str(checkpoint_path),
            "fold":             fold_tag,
        }


# ─── LOSO Cross-Validation Orchestrator ──────────────────────────────────────

def run_loso_cv(
    model_builder: callable,
    splits: list[dict],
    processed_dir: str | Path,
    config: dict[str, Any],
    device: str = "cpu",
    run_name: str = "brave_net",
) -> dict[str, Any]:
    """Run the full Leave-One-Speaker-Out cross-validation experiment.

    Parameters
    ----------
    model_builder : callable() → nn.Module
        Factory function that returns a freshly initialised model.
        Called once per fold to ensure independent training.
    splits : list of dicts from ``dataset.get_loso_splits``.
    processed_dir : Path containing processed image manifests per speaker.
    config : dict
    device : str
    run_name : str

    Returns
    -------
    cv_results : dict with 'per_fold', 'aggregate'
    """
    from ..utils.dataset import BraveNetDataset

    processed_dir = Path(processed_dir)
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]

    per_fold_results = []

    for split in splits:
        test_speaker = split["test_speaker"]
        fold_tag     = f"loso_{test_speaker}"

        # Load manifests for this fold
        train_manifest = processed_dir / f"train_{test_speaker}_manifest.json"
        val_manifest   = processed_dir / f"test_{test_speaker}_manifest.json"

        if not train_manifest.exists() or not val_manifest.exists():
            print(f"[SKIP] Manifests not found for fold {test_speaker}")
            continue

        train_ds = BraveNetDataset(train_manifest, augment=True)
        val_ds   = BraveNetDataset(val_manifest,   augment=False)

        train_labels = [train_ds.manifest[i]["label"] for i in range(len(train_ds))]

        # Weighted sampler for class imbalance
        class_counts = np.bincount(train_labels, minlength=2)
        sample_weights = [1.0 / class_counts[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=(device == "cuda"),
        )

        # Fresh model per fold
        model = model_builder()
        trainer = Trainer(model, config, device=device, run_name=run_name)
        fold_result = trainer.fit(train_loader, val_loader, train_labels, fold_tag)
        per_fold_results.append(fold_result["best_val_metrics"])
        print(f"  ✓ Fold {test_speaker} | F1={fold_result['best_val_metrics'].get('f1', 0):.4f}")

    aggregate = aggregate_loso_results(per_fold_results)
    cv_results = {"per_fold": per_fold_results, "aggregate": aggregate}

    # Save results
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}_loso_results.json"
    with open(results_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"\n[Results saved] → {results_path}")

    # Summary
    print(f"\n{'═'*60}")
    print(f"  LOSO-CV Summary: {run_name}")
    print(f"{'─'*60}")
    for metric, stats in aggregate.items():
        print(f"  {metric:<12}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'═'*60}\n")

    return cv_results
