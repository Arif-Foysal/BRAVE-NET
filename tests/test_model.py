"""
test_model.py
=============
Unit tests for BRAVE-Net and baseline model architectures.

Tests:
  1. Forward pass shape
  2. Gradient flow through all branches
  3. Freezing strategy
  4. Model save and load
  5. Baseline forward passes
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_config():
    return {
        "model": {
            "backbone": "vit_base_patch16_224",
            "pretrained": False,           # No download during CI
            "num_classes": 2,
            "freeze_attention_blocks": True,
            "unfreeze_last_n_blocks": 2,
            "dropout_rate": 0.1,
            "head_hidden_dim": 64,
        }
    }


@pytest.fixture
def dummy_batch():
    """Batch of 4 random RGB images, ViT input size 224×224."""
    return torch.randn(4, 3, 224, 224)


# ─── BRAVE-Net ────────────────────────────────────────────────────────────────

class TestBraveNet:
    def test_forward_output_shape(self, dummy_config, dummy_batch):
        from src.models.brave_net import BraveNet
        model = BraveNet(
            backbone="vit_base_patch16_224",
            pretrained=False,
            num_classes=2,
            freeze_attention_blocks=True,
            unfreeze_last_n_blocks=2,
            dropout_rate=0.1,
            head_hidden_dim=64,
        )
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)
        assert output.shape == (4, 2), f"Expected (4, 2) got {output.shape}"

    def test_gradients_flow(self, dummy_config, dummy_batch):
        """Trainable parameters should receive gradients."""
        from src.models.brave_net import BraveNet
        model = BraveNet(pretrained=False, freeze_attention_blocks=True,
                         unfreeze_last_n_blocks=2, head_hidden_dim=64)
        model.train()
        output = model(dummy_batch)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_frozen_params_no_grad(self, dummy_batch):
        """Frozen parameters should NOT have gradients after backward."""
        from src.models.brave_net import BraveNet
        model = BraveNet(pretrained=False, freeze_attention_blocks=True,
                         unfreeze_last_n_blocks=2, head_hidden_dim=64)
        model.train()
        output = model(dummy_batch)
        output.sum().backward()

        # The first blocks should be frozen
        first_block_params = list(model.backbone.blocks[0].parameters())
        for param in first_block_params:
            assert not param.requires_grad, (
                "First block should be frozen (requires_grad=False)"
            )

    def test_parameter_count(self):
        """Trainable params should be significantly fewer than total."""
        from src.models.brave_net import BraveNet
        model = BraveNet(pretrained=False, freeze_attention_blocks=True,
                         unfreeze_last_n_blocks=2, head_hidden_dim=64)
        counts = model.count_parameters()
        assert counts["trainable"] < counts["total"]
        assert counts["trainable"] > 0

    def test_save_and_load(self, tmp_path, dummy_batch):
        from src.models.brave_net import BraveNet
        model = BraveNet(pretrained=False, head_hidden_dim=64)
        model.eval()
        with torch.no_grad():
            out_before = model(dummy_batch)

        # Save
        ckpt_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Load into fresh instance
        model2 = BraveNet(pretrained=False, head_hidden_dim=64)
        model2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model2.eval()
        with torch.no_grad():
            out_after = model2(dummy_batch)

        torch.testing.assert_close(out_before, out_after)

    def test_get_features_shape(self, dummy_batch):
        from src.models.brave_net import BraveNet
        model = BraveNet(pretrained=False, head_hidden_dim=64)
        model.eval()
        with torch.no_grad():
            features = model.get_features(dummy_batch)
        # ViT-B/16 embed_dim = 768
        assert features.shape == (4, 768)

    def test_unfreeze_all(self, dummy_batch):
        from src.models.brave_net import BraveNet
        model = BraveNet(pretrained=False, freeze_attention_blocks=True,
                         unfreeze_last_n_blocks=2, head_hidden_dim=64)
        model.unfreeze_all()
        for name, param in model.backbone.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be unfrozen"


# ─── Baseline: ResNet-18 ──────────────────────────────────────────────────────

class TestResNet18Baseline:
    def test_forward_shape(self, dummy_batch):
        from src.models.baselines import ResNet18Baseline
        model = ResNet18Baseline(num_classes=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2)

    def test_gradient_flow(self, dummy_batch):
        from src.models.baselines import ResNet18Baseline
        model = ResNet18Baseline(num_classes=2, pretrained=False)
        model.train()
        out = model(dummy_batch)
        out.sum().backward()
        # All params should have gradients (ResNet-18 is not frozen)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


# ─── Baseline: ViT-Raw ────────────────────────────────────────────────────────

class TestViTBaseline:
    def test_forward_shape(self, dummy_batch):
        from src.models.baselines import ViTBaseline
        model = ViTBaseline(pretrained=False, head_hidden_dim=64)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2)

    def test_parameter_count(self):
        from src.models.baselines import ViTBaseline
        model = ViTBaseline(pretrained=False, head_hidden_dim=64)
        counts = model.count_parameters()
        assert counts["trainable"] < counts["total"]


# ─── Baseline: Random Forest ──────────────────────────────────────────────────

class TestRFBaseline:
    def test_fit_predict(self):
        from src.models.baselines import build_rf_classifier
        clf = build_rf_classifier(n_estimators=10)
        X = np.random.randn(50, 78)
        y = np.array([0] * 25 + [1] * 25)
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        assert preds.shape == (50,)
        assert probs.shape == (50, 2)
        assert np.all((probs >= 0) & (probs <= 1))
