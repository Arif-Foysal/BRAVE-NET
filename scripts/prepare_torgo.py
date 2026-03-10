"""
prepare_torgo.py
================
Script to download hints and preprocess the TORGO and MDVR-KCL datasets.

Usage
-----
    python scripts/prepare_torgo.py --dataset torgo \
        --raw_dir data/raw/TORGO \
        --out_dir data/processed/TORGO \
        --config configs/brave_net_config.yaml \
        --apply_restoration

This script:
  1. Scans the raw dataset directory for .wav files
  2. Applies the full BRAVE-Net feature pipeline (Burg restoration + Mel-Spectrogram)
  3. Also generates un-restored Mel-Spectrograms (for ablation baselines)
  4. Saves per-speaker LOSO split manifests as JSON
  5. Extracts MFCC features for the Random Forest baseline

Output structure
----------------
data/processed/TORGO/
├── brave_net/                     ← Burg-restored Mel-Spectrograms
│   ├── train_F01_manifest.json
│   ├── test_F01_manifest.json
│   └── images/
│       ├── 000000_label1.png
│       └── ...
├── raw_mel/                       ← Un-restored Mel-Spectrograms (for ablations)
│   └── ...
└── mfcc/
    └── features.npz               ← MFCC features for RF baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.dataset import scan_torgo_dataset, scan_mdvrkc_dataset, get_loso_splits
from src.features.feature_pipeline import (
    extract_brave_features,
    extract_mfcc_features,
    process_dataset_to_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TORGO or MDVR-KCL for BRAVE-Net")
    parser.add_argument("--dataset",   choices=["torgo", "mdvr_kcl"], default="torgo")
    parser.add_argument("--raw_dir",   required=True, help="Path to raw dataset")
    parser.add_argument("--out_dir",   required=True, help="Output directory for processed data")
    parser.add_argument("--config",    default="configs/brave_net_config.yaml")
    parser.add_argument("--apply_restoration", action="store_true",
                        help="Apply Burg restoration (for BRAVE-Net). "
                             "If omitted, processes raw Mel-Spectrograms (for baselines).")
    parser.add_argument("--n_jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    out_dir = Path(args.out_dir)
    sub_dir = "brave_net" if args.apply_restoration else "raw_mel"
    images_dir = out_dir / sub_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Dataset        : {args.dataset.upper()}")
    print(f"  Raw dir        : {args.raw_dir}")
    print(f"  Output dir     : {out_dir / sub_dir}")
    print(f"  Restoration    : {'YES (BRAVE-Net)' if args.apply_restoration else 'NO (Ablation baseline)'}")
    print(f"{'─'*60}\n")

    # ── 1. Scan dataset ───────────────────────────────────────────────────────
    if args.dataset == "torgo":
        records = scan_torgo_dataset(args.raw_dir)
    else:
        records = scan_mdvrkc_dataset(args.raw_dir)

    if not records:
        print(f"[ERROR] No audio files found in {args.raw_dir}")
        print("  Make sure you have downloaded the dataset and placed it in the correct directory.")
        sys.exit(1)

    print(f"Found {len(records)} audio files.")
    pd_count = sum(1 for r in records if r["label"] == 1)
    hc_count = sum(1 for r in records if r["label"] == 0)
    print(f"  PD (dysarthric): {pd_count} | HC (control): {hc_count}")

    # ── 2. Generate LOSO splits ───────────────────────────────────────────────
    splits = get_loso_splits(records)
    print(f"\nLOSO splits: {len(splits)} speakers")

    for split in splits:
        test_speaker   = split["test_speaker"]
        train_records  = split["train_records"]
        test_records   = split["test_records"]

        # Process training images
        train_img_dir = images_dir / f"train_{test_speaker}"
        print(f"\n  Processing TRAIN split (leave out: {test_speaker}) ...")
        train_manifest = process_dataset_to_images(
            audio_paths=[r["audio_path"] for r in train_records],
            labels=[r["label"] for r in train_records],
            output_dir=train_img_dir,
            config=config,
            apply_restoration=args.apply_restoration,
            n_jobs=args.n_jobs,
        )
        for i, item in enumerate(train_manifest):
            item["speaker_id"] = train_records[i]["speaker_id"]

        # Save train manifest
        train_manifest_path = out_dir / sub_dir / f"train_{test_speaker}_manifest.json"
        with open(train_manifest_path, "w") as f:
            json.dump(train_manifest, f, indent=2)

        # Process test images
        test_img_dir = images_dir / f"test_{test_speaker}"
        print(f"  Processing TEST  split (test speaker: {test_speaker}) ...")
        test_manifest = process_dataset_to_images(
            audio_paths=[r["audio_path"] for r in test_records],
            labels=[r["label"] for r in test_records],
            output_dir=test_img_dir,
            config=config,
            apply_restoration=args.apply_restoration,
            n_jobs=args.n_jobs,
        )
        for i, item in enumerate(test_manifest):
            item["speaker_id"] = test_records[i]["speaker_id"]

        test_manifest_path = out_dir / sub_dir / f"test_{test_speaker}_manifest.json"
        with open(test_manifest_path, "w") as f:
            json.dump(test_manifest, f, indent=2)

        print(f"  ✓ {test_speaker}: {len(train_manifest)} train | {len(test_manifest)} test")

    # ── 3. Extract MFCC features for RF baseline ──────────────────────────────
    if not args.apply_restoration:
        print(f"\nExtracting MFCC features for Random Forest baseline ...")
        mfcc_dir = out_dir / "mfcc"
        mfcc_dir.mkdir(parents=True, exist_ok=True)

        all_features, all_labels, all_speakers = [], [], []
        for rec in tqdm(records, desc="MFCC"):
            feat = extract_mfcc_features(rec["audio_path"], config)
            all_features.append(feat)
            all_labels.append(rec["label"])
            all_speakers.append(rec["speaker_id"])

        np.savez(
            mfcc_dir / "features.npz",
            features=np.array(all_features),
            labels=np.array(all_labels),
            speakers=np.array(all_speakers),
        )
        print(f"  ✓ MFCC features saved → {mfcc_dir / 'features.npz'}")

    print(f"\n[Done] Preprocessing complete → {out_dir / sub_dir}")


if __name__ == "__main__":
    main()
