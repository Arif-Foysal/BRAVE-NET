"""
feature_pipeline.py
===================
Full feature extraction pipeline for BRAVE-Net.

Orchestrates the two-stage transformation:

  Stage 1 — Physics-Informed Signal Restoration
      Raw audio → Burg LP analysis → Residual amplification → Restored signal

  Stage 2 — Visual Feature Representation
      Restored signal → Mel-Spectrogram (128 bands) → 224×224 RGB image

The output of this pipeline is a 3-channel image tensor ready for the
ViT-B/16 backbone.  The same pipeline (without Stage 1) is used for the
ablation baselines that skip restoration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import librosa
import librosa.display
from PIL import Image


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def extract_mel_spectrogram_image(
    signal: NDArray[np.float64],
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    top_db: float = 80.0,
    image_size: tuple[int, int] = (224, 224),
) -> NDArray[np.uint8]:
    """Convert a speech signal to a Mel-Spectrogram image (H × W × 3).

    Parameters
    ----------
    signal : ndarray, shape (N,)
    sample_rate : int
    n_mels : int              Number of Mel filter banks (default 128).
    n_fft : int               FFT window size (default 1024).
    hop_length : int          STFT hop in samples (default 256).
    win_length : int          STFT window length in samples (default 1024).
    f_min, f_max : float      Frequency range (Hz).
    top_db : float            Dynamic range cutoff in dB.
    image_size : (H, W)       Output image size for ViT (default 224×224).

    Returns
    -------
    img_array : ndarray, shape (H, W, 3), dtype uint8
        RGB Mel-Spectrogram image in [0, 255].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import io

    # Compute power Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal.astype(np.float32),
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=top_db)

    # Normalise to [0, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Apply viridis colormap → RGB
    colormap = cm.get_cmap("viridis")
    rgb = (colormap(mel_norm)[:, :, :3] * 255).astype(np.uint8)

    # Resize to target image size (H, W)
    pil_img = Image.fromarray(rgb)
    pil_img = pil_img.resize((image_size[1], image_size[0]), Image.BILINEAR)

    return np.array(pil_img, dtype=np.uint8)


def extract_brave_features(
    audio_path: str | Path,
    config: dict,
    apply_restoration: bool = True,
) -> NDArray[np.uint8]:
    """Full BRAVE-Net feature extraction pipeline for a single audio file.

    Pipeline:
      1. Load & preprocess audio (resample, mono, trim silence)
      2. [Optional] Apply Burg Residual Restoration
      3. Compute Mel-Spectrogram image (H × W × 3)

    Parameters
    ----------
    audio_path : str or Path
        Path to the .wav audio file.
    config : dict
        Loaded YAML configuration (see configs/brave_net_config.yaml).
    apply_restoration : bool
        If True, apply the Burg restoration step (BRAVE-Net).
        If False, skip restoration (for ablation baselines).

    Returns
    -------
    image : ndarray, shape (H, W, 3), dtype uint8
        Mel-Spectrogram image ready for ViT input.
    """
    from .residual_error import restore_full_signal

    audio_cfg = config["audio"]
    mel_cfg   = config["mel_spectrogram"]
    burg_cfg  = config["burg_lp"]
    rest_cfg  = config["restoration"]

    # ── Stage 1: Load & preprocess ───────────────────────────────────────────
    signal, sr = librosa.load(
        str(audio_path),
        sr=audio_cfg["sample_rate"],
        mono=audio_cfg["mono"],
    )
    signal = signal.astype(np.float64)

    # Voice activity detection: trim leading/trailing silence
    signal, _ = librosa.effects.trim(signal, top_db=20)

    if len(signal) == 0:
        # Return blank image for empty/silent recordings
        h, w = mel_cfg["image_size"]
        return np.zeros((h, w, 3), dtype=np.uint8)

    # ── Stage 2 (optional): Burg Restoration ─────────────────────────────────
    if apply_restoration:
        signal = restore_full_signal(
            signal,
            sample_rate=sr,
            lpc_order=burg_cfg["lpc_order"],
            frame_length_ms=burg_cfg["frame_length_ms"],
            hop_length_ms=burg_cfg["hop_length_ms"],
            tremor_band_hz=tuple(rest_cfg["tremor_band_hz"]),
            glottal_band_hz=tuple(rest_cfg["glottal_band_hz"]),
            amplification_alpha=rest_cfg["amplification_alpha"],
            residual_weight=rest_cfg["residual_weight"],
        )

    # ── Stage 3: Mel-Spectrogram image ────────────────────────────────────────
    image = extract_mel_spectrogram_image(
        signal,
        sample_rate=sr,
        n_mels=mel_cfg["n_mels"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg["win_length"],
        f_min=mel_cfg["f_min"],
        f_max=mel_cfg["f_max"],
        top_db=mel_cfg["top_db"],
        image_size=tuple(mel_cfg["image_size"]),
    )
    return image


# ─── MFCC Features (Baseline 1) ───────────────────────────────────────────────

def extract_mfcc_features(
    audio_path: str | Path,
    config: dict,
    n_mfcc: int = 13,
) -> NDArray[np.float64]:
    """Extract MFCC feature vector for the Random Forest baseline.

    Computes 13 MFCCs + delta + delta-delta, then takes the mean and
    standard deviation across frames → feature vector of length 13*3*2=78.

    Parameters
    ----------
    audio_path : str or Path
    config : dict
    n_mfcc : int

    Returns
    -------
    features : ndarray, shape (78,)
    """
    audio_cfg = config["audio"]
    signal, sr = librosa.load(
        str(audio_path),
        sr=audio_cfg["sample_rate"],
        mono=audio_cfg["mono"],
    )
    signal, _ = librosa.effects.trim(signal, top_db=20)
    if len(signal) == 0:
        return np.zeros(n_mfcc * 3 * 2, dtype=np.float64)

    mfccs  = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta1 = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    feature_parts = []
    for feat_matrix in [mfccs, delta1, delta2]:
        feature_parts.append(np.mean(feat_matrix, axis=1))
        feature_parts.append(np.std(feat_matrix, axis=1))

    return np.concatenate(feature_parts).astype(np.float64)


# ─── Batch Processing ─────────────────────────────────────────────────────────

def process_dataset_to_images(
    audio_paths: list[str | Path],
    labels: list[int],
    output_dir: str | Path,
    config: dict,
    apply_restoration: bool = True,
    n_jobs: int = 4,
) -> list[dict]:
    """Process a list of audio files and save Mel-Spectrogram images.

    Parameters
    ----------
    audio_paths : list of paths
    labels : list of int (0=HC, 1=PD)
    output_dir : Path where .png images will be saved
    config : dict
    apply_restoration : bool
    n_jobs : int  Number of parallel workers.

    Returns
    -------
    manifest : list of dicts with keys 'image_path', 'label', 'audio_path'
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _process_one(idx: int, audio_path: str | Path, label: int) -> dict:
        try:
            image = extract_brave_features(
                audio_path, config, apply_restoration=apply_restoration
            )
            img_path = output_dir / f"{idx:06d}_label{label}.png"
            Image.fromarray(image).save(str(img_path))
            return {"image_path": str(img_path), "label": label, "audio_path": str(audio_path)}
        except Exception as exc:
            print(f"[WARNING] Failed to process {audio_path}: {exc}")
            return None

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one)(i, p, l)
        for i, (p, l) in enumerate(tqdm(
            zip(audio_paths, labels),
            total=len(audio_paths),
            desc="Extracting features",
        ))
    )
    return [r for r in results if r is not None]
