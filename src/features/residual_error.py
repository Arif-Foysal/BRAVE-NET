"""
residual_error.py
=================
Residual Error Estimation for the BRAVE-Net pipeline.

After computing LP coefficients via Burg's method, this module:

1.  Computes the **residual (excitation) signal**:
        e[n] = s[n] + a_1*s[n-1] + … + a_p*s[n-p]
    This is the component of speech that the all-pole vocal-tract model
    *cannot* explain — it encodes glottal excitation irregularities.

2.  Performs **Physics-Informed Signal Restoration** (Ohidujjaman, 2025):
    Rather than discarding e[n], the algorithm selectively amplifies the
    residual in frequency bands associated with PD vocal biomarkers:
      • Glottal tremors:  100–300 Hz  (laryngeal flutter)
      • Tremor modulation: 4–8 Hz    (Parkinsonian tremor rate)
    The amplified residual is then fed back through the LP synthesis
    filter to produce a spectrally enhanced waveform.

3.  Extracts **statistical features** from the residual for downstream
    analysis and ablation experiments.

Key insight (research hypothesis)
----------------------------------
Standard denoising attenuates the residual, treating it as noise.
This discards the very biomarkers that distinguish PD from healthy speech.
The Burg residual estimation framework *preserves and enhances* these
components — turning the "noise" into a diagnostic signal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, lfilter


# ─── Residual Signal ─────────────────────────────────────────────────────────

def compute_residual_signal(
    signal: NDArray[np.float64],
    lpc_coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply the LP analysis filter to obtain the residual (excitation) signal.

    Implements:
        E(z) = A(z) · S(z)
    In the time domain:
        e[n] = s[n] + sum_{k=1}^{p} a[k] * s[n-k]

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Speech signal frame.
    lpc_coeffs : ndarray, shape (p,)
        LP coefficients [a_1, …, a_p] from ``compute_burg_lpc``.

    Returns
    -------
    residual : ndarray, shape (N,)
        Residual (excitation) signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    # Analysis filter A(z): numerator = [1, a_1, ..., a_p], denominator = [1]
    a_filter = np.concatenate([[1.0], lpc_coeffs])
    residual = lfilter(a_filter, [1.0], signal)
    return residual


def compute_residual_features(
    signal: NDArray[np.float64],
    lpc_coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract statistical features from the LP residual of one frame.

    Features extracted:
        [0] Mean
        [1] Standard deviation
        [2] Skewness
        [3] Excess kurtosis
        [4] RMS energy
        [5] Zero-crossing rate (normalised by frame length)

    These statistics capture the non-Gaussian, irregular glottal
    excitation patterns characteristic of PD dysphonia.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    lpc_coeffs : ndarray, shape (p,)

    Returns
    -------
    features : ndarray, shape (6,)
    """
    from scipy.stats import skew, kurtosis

    residual = compute_residual_signal(signal, lpc_coeffs)
    N = len(residual)
    if N == 0:
        return np.zeros(6, dtype=np.float64)

    mean_val = float(np.mean(residual))
    std_val  = float(np.std(residual))
    skew_val = float(skew(residual))
    kurt_val = float(kurtosis(residual))          # excess kurtosis (Fisher)
    rms_val  = float(np.sqrt(np.mean(residual ** 2)))
    zcr_val  = float(np.sum(np.abs(np.diff(np.sign(residual)))) / (2 * N))

    return np.array(
        [mean_val, std_val, skew_val, kurt_val, rms_val, zcr_val],
        dtype=np.float64,
    )


# ─── Physics-Informed Signal Restoration ─────────────────────────────────────

def restore_speech_burg(
    signal: NDArray[np.float64],
    lpc_coeffs: NDArray[np.float64],
    sample_rate: int,
    tremor_band_hz: tuple[float, float] = (4.0, 8.0),
    glottal_band_hz: tuple[float, float] = (100.0, 300.0),
    amplification_alpha: float = 1.5,
    residual_weight: float = 0.4,
) -> NDArray[np.float64]:
    """Apply the Burg-based signal restoration method (Ohidujjaman et al., 2025).

    This is the core **"Signal-First"** transformation that distinguishes
    BRAVE-Net from standard deep learning approaches.

    Algorithm:
    1.  Compute residual: e[n] = A(z) · s[n]
    2.  Bandpass-filter e[n] to retain:
          a) Glottal source band (100–300 Hz) — captures vocal tract dynamics
          b) Tremor modulation band (4–8 Hz) — captures Parkinsonian tremor
    3.  Amplify filtered residual by ``amplification_alpha``
    4.  Reconstruct via LP synthesis:
          s_restored[n] = LP_synthesis(e_amplified[n], lpc_coeffs)
    5.  Blend with original: output = (1-w)*s + w*s_restored

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input speech frame (float64).
    lpc_coeffs : ndarray, shape (p,)
        Burg LP coefficients for this frame.
    sample_rate : int
        Audio sample rate in Hz.
    tremor_band_hz : tuple[float, float]
        Low/high cut for tremor modulation bandpass (Hz).
    glottal_band_hz : tuple[float, float]
        Low/high cut for glottal source bandpass (Hz).
    amplification_alpha : float
        Amplification factor applied to the PD-relevant residual components.
    residual_weight : float
        Blending weight w ∈ [0,1]; controls how much of the restored signal
        is mixed with the original.

    Returns
    -------
    restored : ndarray, shape (N,)
        Spectrally enhanced speech frame.
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyq = sample_rate / 2.0

    # 1. Compute residual (excitation signal)
    residual = compute_residual_signal(signal, lpc_coeffs)

    # 2. Extract PD-relevant components from the residual
    pd_residual = np.zeros_like(residual)

    # 2a. Glottal source band (100–300 Hz) — vocal tract irregularities
    if glottal_band_hz[1] < nyq and glottal_band_hz[0] > 0:
        b, a = butter(
            4,
            [glottal_band_hz[0] / nyq, glottal_band_hz[1] / nyq],
            btype="band",
        )
        pd_residual += filtfilt(b, a, residual)

    # 2b. Tremor modulation band (4–8 Hz) — Parkinsonian oscillation
    if tremor_band_hz[1] < nyq and tremor_band_hz[0] > 0:
        b, a = butter(
            4,
            [tremor_band_hz[0] / nyq, tremor_band_hz[1] / nyq],
            btype="band",
        )
        pd_residual += filtfilt(b, a, residual)

    # 3. Amplify the PD-relevant residual components
    amplified_residual = residual + amplification_alpha * pd_residual

    # 4. Reconstruct speech through LP synthesis filter 1/A(z)
    a_filter = np.concatenate([[1.0], lpc_coeffs])
    s_restored = lfilter([1.0], a_filter, amplified_residual)

    # 5. Energy normalisation (preserve original RMS)
    orig_rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
    rest_rms = np.sqrt(np.mean(s_restored ** 2)) + 1e-10
    s_restored = s_restored * (orig_rms / rest_rms)

    # 6. Blend original + restored
    output = (1.0 - residual_weight) * signal + residual_weight * s_restored
    return output.astype(np.float64)


def restore_full_signal(
    signal: NDArray[np.float64],
    sample_rate: int,
    lpc_order: int = 14,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    tremor_band_hz: tuple[float, float] = (4.0, 8.0),
    glottal_band_hz: tuple[float, float] = (100.0, 300.0),
    amplification_alpha: float = 1.5,
    residual_weight: float = 0.4,
) -> NDArray[np.float64]:
    """Apply Burg restoration frame-by-frame to a full speech utterance.

    Uses overlap-add (OLA) to reconstruct the enhanced signal from
    individually restored frames.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Full utterance (already resampled to ``sample_rate``).
    sample_rate : int
    lpc_order : int
    frame_length_ms, hop_length_ms : float
    tremor_band_hz, glottal_band_hz : tuple[float, float]
    amplification_alpha, residual_weight : float

    Returns
    -------
    restored_signal : ndarray, shape (N,)
        Restored utterance, same length as input.
    """
    from .burg_lp import frame_signal, compute_burg_lpc
    from scipy.signal import get_window

    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    frame_len = int(sample_rate * frame_length_ms / 1000)
    hop_len   = int(sample_rate * hop_length_ms  / 1000)

    # Pad signal to fit integer number of frames
    pad_len = frame_len + hop_len * ((N - frame_len) // hop_len + 1)
    padded  = np.pad(signal, (0, max(0, pad_len - N)))

    # Overlap-add accumulators
    output_signal = np.zeros(len(padded), dtype=np.float64)
    weight_signal = np.zeros(len(padded), dtype=np.float64)
    window = get_window("hann", frame_len)

    num_frames = 1 + (N - frame_len) // hop_len
    for i in range(num_frames):
        start = i * hop_len
        end   = start + frame_len
        frame = padded[start:end].copy()

        if np.all(np.abs(frame) < 1e-8):
            # Silent frame — skip restoration
            output_signal[start:end] += frame * window
        else:
            lpc, _, _ = compute_burg_lpc(frame, order=lpc_order)
            restored_frame = restore_speech_burg(
                frame,
                lpc,
                sample_rate,
                tremor_band_hz=tremor_band_hz,
                glottal_band_hz=glottal_band_hz,
                amplification_alpha=amplification_alpha,
                residual_weight=residual_weight,
            )
            output_signal[start:end] += restored_frame * window

        weight_signal[start:end] += window

    # Normalise by overlap weight
    mask = weight_signal > 1e-8
    output_signal[mask] /= weight_signal[mask]

    return output_signal[:N]


# ─── Frame-Level Residual Features ───────────────────────────────────────────

def compute_frame_residual_features(
    signal: NDArray[np.float64],
    lpc_coeffs_per_frame: NDArray[np.float64],
    frame_length: int,
    hop_length: int,
) -> NDArray[np.float64]:
    """Extract residual statistical features for each frame.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    lpc_coeffs_per_frame : ndarray, shape (num_frames, lpc_order)
    frame_length : int
    hop_length : int

    Returns
    -------
    residual_features : ndarray, shape (num_frames, 6)
        Per-frame residual feature vectors.
    """
    from .burg_lp import frame_signal as _frame_signal

    frames = _frame_signal(signal, frame_length, hop_length, window="none")
    num_frames = min(len(frames), len(lpc_coeffs_per_frame))
    features = np.zeros((num_frames, 6), dtype=np.float64)

    for i in range(num_frames):
        features[i] = compute_residual_features(
            frames[i], lpc_coeffs_per_frame[i]
        )

    return features
