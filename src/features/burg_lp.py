"""
burg_lp.py
==========
Burg's Method for Linear Predictive Coding (LPC).

Implementation of the Burg algorithm as described in:
  Ohidujjaman et al. (2025), "Restoration of speech packet estimating
  residue of Burg linear prediction", Results in Engineering.

The Burg method estimates Linear Prediction Coefficients (LPCs) by
minimising the sum of squared forward and backward prediction errors
using the Levinson–Durbin recursion, without requiring an explicit
autocorrelation step. This yields numerically stable (minimum-phase)
all-pole filters, making it well suited for short speech frames.

Mathematical Formulation
------------------------
For a signal s[n] of length N and LP order p:

  Forward  prediction error at order m:
      f_m[n] = s[n] + sum_{k=1}^{m} a_m[k] * s[n-k]

  Backward prediction error at order m:
      b_m[n] = s[n-m] + sum_{k=1}^{m} a_m[k] * s[n-m+k]

  Burg reflection coefficient at order m:
               -2 * sum_{n=m}^{N-1} f_{m-1}[n] * b_{m-1}[n-1]
      k_m = ─────────────────────────────────────────────────────
             sum_{n=m}^{N-1} (f_{m-1}[n]^2 + b_{m-1}[n-1]^2)

  Levinson–Durbin update:
      a_m[i] = a_{m-1}[i] + k_m * a_{m-1}[m-i]  for i = 1 … m-1
      a_m[m] = k_m

  Prediction error power update:
      E_m = (1 - k_m^2) * E_{m-1}
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_burg_lpc(
    signal: NDArray[np.float64],
    order: int = 14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Estimate Linear Prediction Coefficients via Burg's method.

    Parameters
    ----------
    signal : array_like, shape (N,)
        Single-channel speech frame (float64 or float32).
    order : int
        LP model order ``p``.  Typical values for speech: 10–16.
        Must satisfy ``1 <= order < len(signal)``.

    Returns
    -------
    lpc_coeffs : ndarray, shape (order,)
        LP coefficients [a_1, a_2, …, a_p].
        The full error filter is A(z) = 1 + a_1 z^{-1} + … + a_p z^{-p}.
    reflection_coeffs : ndarray, shape (order,)
        Reflection (PARCOR) coefficients [k_1, k_2, …, k_p].
        Stability guarantee: |k_m| < 1 for all m.
    prediction_error : float
        Normalised prediction error power after order ``p`` iterations.

    Raises
    ------
    ValueError
        If ``order`` is out of range or the signal contains NaN/Inf.

    Notes
    -----
    The returned coefficients follow the convention used in the Ohidujjaman
    (2025) paper: coefficients are the *predictor* weights (positive sign),
    so the prediction of s[n] is:
        s_hat[n] = -sum_{k=1}^{p} lpc_coeffs[k-1] * s[n-k]
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    _validate_signal(signal, order)

    N = len(signal)
    p = order

    # Initialise forward/backward error vectors
    f = signal.copy()   # forward  prediction error
    b = signal.copy()   # backward prediction error

    lpc     = np.zeros(p, dtype=np.float64)   # current LP coefficients
    refls   = np.zeros(p, dtype=np.float64)   # reflection coefficients
    error   = float(np.dot(signal, signal) / N)  # initial error power

    for m in range(p):
        # ── Compute Burg reflection coefficient ──────────────────────────
        # Use relative indexing: after each iteration f and b are shortened
        # by one element, so f[1:] and b[:-1] always have matching lengths.
        numer = -2.0 * np.dot(f[1:], b[:-1])
        denom =  np.dot(f[1:], f[1:]) + np.dot(b[:-1], b[:-1])

        if denom < 1e-12:
            # Degenerate frame (silence or near-silence) — stop early
            break

        k_m = numer / denom

        # Clamp to maintain filter stability (|k| must be < 1)
        k_m = float(np.clip(k_m, -1.0 + 1e-9, 1.0 - 1e-9))
        refls[m] = k_m

        # ── Levinson–Durbin update ────────────────────────────────────────
        # Update existing coefficients
        lpc_new = lpc.copy()
        lpc_new[:m] = lpc[:m] + k_m * lpc[m - 1:: -1][:m]
        lpc_new[m]  = k_m
        lpc = lpc_new

        # ── Update error vectors ──────────────────────────────────────────
        f_new = f[1:] + k_m * b[:-1]
        b_new = b[:-1] + k_m * f[1:]
        f = f_new
        b = b_new

        # ── Update prediction error power ─────────────────────────────────
        error *= 1.0 - k_m ** 2

    return lpc, refls, error


def lpc_to_lpcc(
    lpc_coeffs: NDArray[np.float64],
    gain: float,
    num_cepstral: int = 13,
) -> NDArray[np.float64]:
    """Convert LP coefficients to LP Cepstral Coefficients (LPCC).

    Uses the standard recursion (Markel & Gray, 1976):

        c[0] = log(gain)
        c[m] = -a[m] - sum_{k=1}^{m-1} (k/m) * c[k] * a[m-k]   (1 ≤ m ≤ p)
        c[m] = -sum_{k=m-p}^{m-1} (k/m) * c[k] * a[m-k]          (m > p)

    Parameters
    ----------
    lpc_coeffs : ndarray, shape (p,)
        LP coefficients from ``compute_burg_lpc``.
    gain : float
        Prediction error power (returned as third value of ``compute_burg_lpc``).
    num_cepstral : int
        Number of cepstral coefficients to compute (default 13).

    Returns
    -------
    lpcc : ndarray, shape (num_cepstral,)
        LP Cepstral Coefficients c[0], c[1], …, c[num_cepstral-1].
    """
    p = len(lpc_coeffs)
    # Pad with zeros if more coefficients requested than LP order
    c = np.zeros(num_cepstral, dtype=np.float64)
    a = np.concatenate([[0.0], lpc_coeffs])  # 1-indexed: a[1..p]

    c[0] = np.log(max(gain, 1e-10))

    for m in range(1, num_cepstral):
        if m <= p:
            c[m] = -a[m] - sum(
                (k / m) * c[k] * a[m - k]
                for k in range(1, m)
                if m - k <= p
            )
        else:
            c[m] = -sum(
                (k / m) * c[k] * a[m - k]
                for k in range(max(1, m - p), m)
                if m - k <= p
            )

    return c


def synthesise_speech(
    residual: NDArray[np.float64],
    lpc_coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reconstruct speech from a residual (excitation) signal and LP coefficients.

    Implements the all-pole synthesis filter:
        S(z) = E(z) / A(z)   where A(z) = 1 + sum_{k=1}^{p} a[k] z^{-k}

    Equivalent to:  s[n] = e[n] - sum_{k=1}^{p} a[k] * s[n-k]

    Parameters
    ----------
    residual : ndarray, shape (N,)
        Excitation (residual error) signal.
    lpc_coeffs : ndarray, shape (p,)
        LP coefficients [a_1, …, a_p].

    Returns
    -------
    speech : ndarray, shape (N,)
        Synthesised speech signal.
    """
    from scipy.signal import lfilter
    # A(z) numerator = [1, a_1, ..., a_p], denominator = [1]
    a_filter = np.concatenate([[1.0], lpc_coeffs])
    speech = lfilter([1.0], a_filter, residual)
    return speech


# ─── Frame-Level Processing ───────────────────────────────────────────────────

def frame_signal(
    signal: NDArray[np.float64],
    frame_length: int,
    hop_length: int,
    window: str = "hamming",
) -> NDArray[np.float64]:
    """Split a signal into overlapping frames with optional windowing.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    frame_length : int   Number of samples per frame.
    hop_length : int     Number of samples between successive frames.
    window : str         Scipy window name ('hamming', 'hann', etc.) or 'none'.

    Returns
    -------
    frames : ndarray, shape (num_frames, frame_length)
    """
    from scipy.signal import get_window

    N = len(signal)
    num_frames = 1 + (N - frame_length) // hop_length
    indices = (
        np.arange(frame_length)[None, :]
        + np.arange(num_frames)[:, None] * hop_length
    )
    frames = signal[indices].astype(np.float64)

    if window.lower() != "none":
        win = get_window(window, frame_length)
        frames *= win[None, :]

    return frames


def compute_burg_features_per_frame(
    signal: NDArray[np.float64],
    sample_rate: int,
    lpc_order: int = 14,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    num_cepstral: int = 13,
) -> dict[str, NDArray[np.float64]]:
    """Extract per-frame Burg LP features from a speech signal.

    For every frame computes:
    - LPC coefficients  (shape: num_frames × lpc_order)
    - Reflection coefficients (shape: num_frames × lpc_order)
    - LPCC coefficients (shape: num_frames × num_cepstral)
    - Prediction error power (shape: num_frames,)

    Parameters
    ----------
    signal : ndarray, shape (N,)
    sample_rate : int
    lpc_order : int
    frame_length_ms, hop_length_ms : float
    num_cepstral : int  Number of LPCC coefficients to compute.

    Returns
    -------
    dict with keys: 'lpc', 'reflection', 'lpcc', 'error'
    """
    frame_len = int(sample_rate * frame_length_ms / 1000)
    hop_len   = int(sample_rate * hop_length_ms  / 1000)

    frames = frame_signal(signal, frame_len, hop_len)
    num_frames = len(frames)

    lpc_mat   = np.zeros((num_frames, lpc_order),    dtype=np.float64)
    refl_mat  = np.zeros((num_frames, lpc_order),    dtype=np.float64)
    lpcc_mat  = np.zeros((num_frames, num_cepstral), dtype=np.float64)
    error_vec = np.zeros(num_frames,                 dtype=np.float64)

    for i, frame in enumerate(frames):
        lpc, refl, err = compute_burg_lpc(frame, order=lpc_order)
        lpc_mat[i]   = lpc
        refl_mat[i]  = refl
        error_vec[i] = err
        lpcc_mat[i]  = lpc_to_lpcc(lpc, err, num_cepstral)

    return {
        "lpc":        lpc_mat,
        "reflection": refl_mat,
        "lpcc":       lpcc_mat,
        "error":      error_vec,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _validate_signal(signal: NDArray[np.float64], order: int) -> None:
    if signal.ndim != 1:
        raise ValueError(f"Signal must be 1-D, got shape {signal.shape}.")
    if len(signal) < order + 1:
        raise ValueError(
            f"Signal length ({len(signal)}) must be > LP order ({order})."
        )
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or Inf values.")
    if order < 1:
        raise ValueError(f"LP order must be >= 1, got {order}.")
