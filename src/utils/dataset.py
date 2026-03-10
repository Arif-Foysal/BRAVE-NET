"""
dataset.py
==========
PyTorch Dataset classes for TORGO and MDVR-KCL.

TORGO Directory Structure (expected)
-------------------------------------
data/raw/TORGO/
├── F01/         ← Dysarthric female speaker
│   ├── Session1/
│   │   ├── wav_arrayMic/
│   │   │   ├── 0001.wav
│   │   │   └── ...
│   │   └── wav_headMic/
├── F03/
├── M01/  ...
├── FC1/         ← Female control
├── FC2/  ...
└── MC1/  ...

Label convention: PD (dysarthric) = 1, HC (healthy control) = 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ─── Torchvision Transform ────────────────────────────────────────────────────

def get_vit_transforms(
    image_size: int = 224,
    augment: bool = False,
) -> transforms.Compose:
    """Standard ImageNet normalisation + optional SpecAugment-style augmentation.

    Parameters
    ----------
    image_size : int
    augment : bool   If True, add random horizontal flip + erase.

    Returns
    -------
    transform : torchvision.transforms.Compose
    """
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225],
        ),
    ]
    if augment:
        base = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ]
    return transforms.Compose(base)


# ─── Manifest-Based Dataset ───────────────────────────────────────────────────

class BraveNetDataset(Dataset):
    """Image-based dataset for BRAVE-Net.

    Reads a JSON manifest produced by ``feature_pipeline.process_dataset_to_images``.

    Each item:
        image_tensor : FloatTensor, shape (3, H, W)   — normalised
        label        : LongTensor, scalar             — 0=HC, 1=PD
        meta         : dict with 'audio_path', 'speaker_id'

    Parameters
    ----------
    manifest_path : str or Path   Path to JSON manifest.
    transform : callable, optional   Override default ViT transform.
    augment : bool                   Enable training augmentation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path) as f:
            self.manifest: list[dict] = json.load(f)

        self.transform = transform or get_vit_transforms(augment=augment)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        item = self.manifest[idx]
        img_path = item["image_path"]
        label    = int(item["label"])

        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)

        meta = {
            "audio_path": item.get("audio_path", ""),
            "speaker_id": item.get("speaker_id", ""),
        }
        return tensor, torch.tensor(label, dtype=torch.long), meta


# ─── TORGO Dataset Builder ────────────────────────────────────────────────────

TORGO_DYSARTHRIC = ["F01", "F03", "M01", "M02", "M04", "M05"]
TORGO_CONTROL    = ["FC1", "FC2", "FC3", "MC1", "MC2", "MC3", "MC4"]


def scan_torgo_dataset(
    root: str | Path,
    mic: str = "headMic",
    extensions: tuple[str, ...] = (".wav",),
) -> list[dict]:
    """Walk the TORGO directory tree and collect all audio file paths.

    Parameters
    ----------
    root : Path to TORGO root directory.
    mic : str   Microphone channel to use: 'headMic' or 'arrayMic'.
    extensions : tuple of file suffixes to include.

    Returns
    -------
    records : list of dicts with keys:
        'audio_path', 'speaker_id', 'label' (0=HC, 1=PD),
        'session', 'group' ('dysarthric' or 'control')
    """
    root = Path(root)
    records: list[dict] = []

    all_speakers = (
        [(s, 1, "dysarthric") for s in TORGO_DYSARTHRIC]
        + [(s, 0, "control")  for s in TORGO_CONTROL]
    )

    for speaker_id, label, group in all_speakers:
        speaker_dir = root / speaker_id
        if not speaker_dir.exists():
            continue

        for wav_file in sorted(speaker_dir.rglob(f"wav_{mic}/*")):
            if wav_file.suffix.lower() not in extensions:
                continue
            records.append({
                "audio_path": str(wav_file),
                "speaker_id": speaker_id,
                "label":      label,
                "session":    wav_file.parents[1].name,
                "group":      group,
            })

    return records


def get_loso_splits(
    records: list[dict],
) -> list[dict]:
    """Generate Leave-One-Speaker-Out (LOSO) cross-validation splits.

    Critically, speaker identity is held out to prevent data leakage —
    all utterances from the test speaker are unseen during training.

    Parameters
    ----------
    records : list of dicts from ``scan_torgo_dataset``.

    Returns
    -------
    splits : list of dicts, one per speaker, each with:
        'test_speaker'  : str
        'train_records' : list[dict]
        'test_records'  : list[dict]
    """
    speakers = sorted({r["speaker_id"] for r in records})
    splits = []
    for test_speaker in speakers:
        train = [r for r in records if r["speaker_id"] != test_speaker]
        test  = [r for r in records if r["speaker_id"] == test_speaker]
        splits.append({
            "test_speaker":  test_speaker,
            "train_records": train,
            "test_records":  test,
        })
    return splits


# ─── MDVR-KCL Dataset Builder ─────────────────────────────────────────────────

def scan_mdvrkc_dataset(
    root: str | Path,
    extensions: tuple[str, ...] = (".wav",),
) -> list[dict]:
    """Scan MDVR-KCL dataset.

    Expected structure:
        data/raw/MDVR-KCL/
        ├── PD/
        │   ├── speaker_001/
        │   │   └── *.wav
        └── HC/
            ├── speaker_101/
            │   └── *.wav

    Returns
    -------
    records : list of dicts with 'audio_path', 'speaker_id', 'label'
    """
    root = Path(root)
    records: list[dict] = []

    label_map = {"PD": 1, "HC": 0}
    for group_dir in sorted(root.iterdir()):
        if group_dir.name not in label_map:
            continue
        label = label_map[group_dir.name]
        for speaker_dir in sorted(group_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for wav_file in sorted(speaker_dir.rglob("*")):
                if wav_file.suffix.lower() not in extensions:
                    continue
                records.append({
                    "audio_path": str(wav_file),
                    "speaker_id": f"MDVR_{speaker_dir.name}",
                    "label":      label,
                    "group":      group_dir.name,
                })
    return records
