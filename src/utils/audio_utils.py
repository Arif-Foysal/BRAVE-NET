"""
audio_utils.py
==============
Audio loading, preprocessing, and Voice Activity Detection utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import librosa
import soundfile as sf


def load_audio(
    path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[NDArray[np.float32], int]:
    """Load and resample an audio file.

    Parameters
    ----------
    path : str or Path
    sample_rate : int   Target sample rate.
    mono : bool         Convert to mono.

    Returns
    -------
    signal : ndarray, shape (N,), dtype float32
    sample_rate : int
    """
    signal, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
    return signal.astype(np.float32), sr


def trim_silence(
    signal: NDArray[np.float32],
    top_db: float = 20.0,
) -> NDArray[np.float32]:
    """Trim leading/trailing silence using energy thresholding.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    top_db : float    Threshold in dB below the peak for silence detection.

    Returns
    -------
    trimmed : ndarray
    """
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed


def normalise_loudness(
    signal: NDArray[np.float32],
    target_rms: float = 0.1,
) -> NDArray[np.float32]:
    """Normalise signal to a target RMS level.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    target_rms : float

    Returns
    -------
    normalised : ndarray
    """
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-10:
        return signal
    return signal * (target_rms / rms)


def preprocess_audio(
    path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
    trim_db: float = 20.0,
    normalise: bool = True,
    target_rms: float = 0.1,
) -> tuple[NDArray[np.float32], int]:
    """Full preprocessing: load → resample → trim → normalise.

    Parameters
    ----------
    path : str or Path
    sample_rate, mono, trim_db, normalise, target_rms : see sub-functions.

    Returns
    -------
    signal : ndarray, shape (N,), dtype float32
    sample_rate : int
    """
    signal, sr = load_audio(path, sample_rate=sample_rate, mono=mono)
    signal = trim_silence(signal, top_db=trim_db)
    if normalise and len(signal) > 0:
        signal = normalise_loudness(signal, target_rms=target_rms)
    return signal, sr


def compute_snr(
    original: NDArray[np.float64],
    restored: NDArray[np.float64],
) -> float:
    """Compute Signal-to-Noise Ratio improvement after restoration.

    SNR = 10 * log10( E[s^2] / E[(s-s_hat)^2] )

    Parameters
    ----------
    original : ndarray   Original signal.
    restored : ndarray   Restored/processed signal.

    Returns
    -------
    snr_db : float
    """
    noise = original - restored
    signal_power = np.mean(original ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def compute_spectral_flatness_delta(
    original: NDArray[np.float64],
    restored: NDArray[np.float64],
    n_fft: int = 1024,
) -> float:
    """Measure change in spectral flatness after restoration.

    A decrease in spectral flatness after restoration indicates the signal
    has become more tonal (less noise-like), verifying the restoration.

    Returns
    -------
    delta_sf : float   Spectral flatness of original minus restored.
                       Positive means restoration reduced noise-like character.
    """
    def _spectral_flatness(sig: NDArray) -> float:
        spectrum = np.abs(np.fft.rfft(sig, n=n_fft)) + 1e-10
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        return float(geo_mean / arith_mean)

    return _spectral_flatness(original) - _spectral_flatness(restored)
