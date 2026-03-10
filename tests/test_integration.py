"""
test_integration.py
===================
End-to-end integration test for the full BRAVE-Net pipeline.

Generates synthetic PD and HC voice signals, runs the complete pipeline:
    synthetic audio → Burg restoration → Mel-Spectrogram → Model → prediction

Does NOT require real dataset files.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── Synthetic Signal Generation ─────────────────────────────────────────────

def generate_pd_signal(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    f0: float = 120.0,
    tremor_hz: float = 6.0,
    tremor_depth: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic PD voice: harmonics + 4–8 Hz tremor modulation."""
    rng = np.random.default_rng(seed)
    t   = np.arange(int(duration_s * sample_rate)) / sample_rate

    # Glottal source: sum of harmonics
    signal = sum(
        (1.0 / (k + 1)) * np.sin(2 * np.pi * f0 * (k + 1) * t)
        for k in range(8)
    )
    # Parkinsonian tremor modulation (4–8 Hz)
    tremor = tremor_depth * np.sin(2 * np.pi * tremor_hz * t)
    signal = signal * (1.0 + tremor)
    # Normalise
    signal /= np.abs(signal).max() + 1e-8
    return signal.astype(np.float64)


def generate_hc_signal(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    f0: float = 130.0,
    seed: int = 1,
) -> np.ndarray:
    """Synthetic HC voice: steady harmonics, minimal tremor."""
    rng = np.random.default_rng(seed)
    t   = np.arange(int(duration_s * sample_rate)) / sample_rate

    signal = sum(
        (1.0 / (k + 1)) * np.sin(2 * np.pi * f0 * (k + 1) * t)
        for k in range(8)
    )
    signal /= np.abs(signal).max() + 1e-8
    return signal.astype(np.float64)


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestBurgRestorationPipeline:
    """Test the signal restoration component on synthetic signals."""

    def test_restoration_runs_without_error(self):
        from src.features.residual_error import restore_full_signal
        signal = generate_pd_signal()
        restored = restore_full_signal(signal, sample_rate=16000, lpc_order=14)
        assert restored.shape == signal.shape
        assert np.all(np.isfinite(restored))

    def test_restoration_same_length(self):
        from src.features.residual_error import restore_full_signal
        signal = generate_hc_signal()
        restored = restore_full_signal(signal, sample_rate=16000, lpc_order=14)
        assert len(restored) == len(signal)

    def test_restoration_changes_signal(self):
        """Restored signal should differ from raw signal."""
        from src.features.residual_error import restore_full_signal
        signal = generate_pd_signal()
        restored = restore_full_signal(signal, sample_rate=16000, lpc_order=14)
        diff = np.abs(signal - restored).mean()
        assert diff > 1e-6, "Restoration should modify the signal"

    def test_residual_is_finite(self):
        from src.features.burg_lp import compute_burg_lpc
        from src.features.residual_error import compute_residual_signal
        signal = generate_pd_signal()[:1024]
        lpc, _, _ = compute_burg_lpc(signal, order=14)
        residual = compute_residual_signal(signal, lpc)
        assert np.all(np.isfinite(residual))


class TestMelSpectrogramPipeline:
    """Test the Mel-Spectrogram extraction on synthetic signals."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        return {
            "audio": {"sample_rate": 16000, "mono": True},
            "burg_lp": {"lpc_order": 14, "frame_length_ms": 25, "hop_length_ms": 10},
            "restoration": {
                "tremor_band_hz": [4, 8],
                "glottal_band_hz": [100, 300],
                "amplification_alpha": 1.5,
                "residual_weight": 0.4,
            },
            "mel_spectrogram": {
                "n_mels": 128, "n_fft": 1024, "hop_length": 256,
                "win_length": 1024, "f_min": 0, "f_max": 8000,
                "top_db": 80, "image_size": [224, 224],
            },
        }

    def test_mel_image_shape(self, minimal_config, tmp_path):
        import soundfile as sf
        from src.features.feature_pipeline import extract_brave_features

        # Write synthetic audio to temp file
        signal = generate_pd_signal()
        wav_path = tmp_path / "test_pd.wav"
        sf.write(str(wav_path), signal, 16000)

        image = extract_brave_features(str(wav_path), minimal_config, apply_restoration=False)
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8

    def test_mel_image_restored_shape(self, minimal_config, tmp_path):
        import soundfile as sf
        from src.features.feature_pipeline import extract_brave_features

        signal = generate_pd_signal()
        wav_path = tmp_path / "test_pd_restored.wav"
        sf.write(str(wav_path), signal, 16000)

        image = extract_brave_features(str(wav_path), minimal_config, apply_restoration=True)
        assert image.shape == (224, 224, 3)

    def test_mel_image_range(self, minimal_config, tmp_path):
        import soundfile as sf
        from src.features.feature_pipeline import extract_brave_features

        signal = generate_hc_signal()
        wav_path = tmp_path / "test_hc.wav"
        sf.write(str(wav_path), signal, 16000)

        image = extract_brave_features(str(wav_path), minimal_config, apply_restoration=False)
        assert image.min() >= 0
        assert image.max() <= 255

    def test_restored_image_differs_from_raw(self, minimal_config, tmp_path):
        """Restored Mel-Spectrogram should differ from unrestored."""
        import soundfile as sf
        from src.features.feature_pipeline import extract_brave_features

        signal = generate_pd_signal()
        wav_path = tmp_path / "test_pd_diff.wav"
        sf.write(str(wav_path), signal, 16000)

        raw_img      = extract_brave_features(str(wav_path), minimal_config, apply_restoration=False)
        restored_img = extract_brave_features(str(wav_path), minimal_config, apply_restoration=True)

        diff = np.abs(raw_img.astype(float) - restored_img.astype(float)).mean()
        assert diff > 0.01, "Restored image should differ from raw image"


class TestFullPipeline:
    """End-to-end: synthetic audio → ViT model → probability in [0, 1]."""

    def test_full_pipeline_output_range(self, tmp_path):
        import soundfile as sf
        from src.features.feature_pipeline import extract_brave_features
        from src.models.brave_net import BraveNet
        from src.utils.dataset import get_vit_transforms
        from PIL import Image

        config = {
            "audio": {"sample_rate": 16000, "mono": True},
            "burg_lp": {"lpc_order": 14, "frame_length_ms": 25, "hop_length_ms": 10},
            "restoration": {
                "tremor_band_hz": [4, 8], "glottal_band_hz": [100, 300],
                "amplification_alpha": 1.5, "residual_weight": 0.4,
            },
            "mel_spectrogram": {
                "n_mels": 128, "n_fft": 1024, "hop_length": 256,
                "win_length": 1024, "f_min": 0, "f_max": 8000,
                "top_db": 80, "image_size": [224, 224],
            },
        }

        # Write both PD and HC signals
        for label, gen_fn in [(1, generate_pd_signal), (0, generate_hc_signal)]:
            signal   = gen_fn()
            wav_path = tmp_path / f"signal_label{label}.wav"
            sf.write(str(wav_path), signal, 16000)

            # Extract features (restored)
            image_np = extract_brave_features(str(wav_path), config, apply_restoration=True)

            # Convert to tensor
            transform  = get_vit_transforms(image_size=224, augment=False)
            pil_img    = Image.fromarray(image_np)
            img_tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 224, 224)

            # Run model (no pretrained weights for speed)
            model = BraveNet(
                pretrained=False,
                freeze_attention_blocks=False,
                head_hidden_dim=64,
            )
            model.eval()
            with torch.no_grad():
                logits = model(img_tensor)
                prob   = torch.softmax(logits, dim=1)[0, 1].item()

            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range [0, 1]"
            assert logits.shape == (1, 2)
