"""
Tests for src/networks.py.

Validates ContinuationNet architecture and train_network training routine.
"""

import numpy as np
import torch
import pytest
from src.networks import ContinuationNet, train_network


# ─────────────────────────────────────────────────────────────────────────────
# ContinuationNet
# ─────────────────────────────────────────────────────────────────────────────

class TestContinuationNet:
    def test_output_shape_1d(self):
        """1-D input → scalar output per sample."""
        net = ContinuationNet(input_dim=1, hidden_width=16, depth=2)
        x = torch.randn(32, 1)
        out = net(x)
        assert out.shape == (32,)

    def test_output_shape_5d(self):
        """5-D input → scalar output per sample."""
        net = ContinuationNet(input_dim=5, hidden_width=32, depth=3)
        x = torch.randn(64, 5)
        out = net(x)
        assert out.shape == (64,)

    def test_output_scalar_single_sample(self):
        """Single sample returns a 1-D tensor of length 1."""
        net = ContinuationNet(input_dim=2, hidden_width=8, depth=2)
        x = torch.randn(1, 2)
        out = net(x)
        assert out.shape == (1,)

    def test_all_finite_outputs(self):
        """Network output is always finite for reasonable inputs."""
        net = ContinuationNet(input_dim=3, hidden_width=32, depth=4)
        x = torch.randn(100, 3)
        out = net(x)
        assert torch.all(torch.isfinite(out))

    def test_depth_2_minimum(self):
        """Network with depth=2 (minimum) has valid architecture."""
        net = ContinuationNet(input_dim=1, hidden_width=8, depth=2)
        x = torch.randn(10, 1)
        out = net(x)
        assert out.shape == (10,)

    def test_depth_5(self):
        """Deeper network (depth=5) still runs correctly."""
        net = ContinuationNet(input_dim=2, hidden_width=64, depth=5)
        x = torch.randn(50, 2)
        out = net(x)
        assert out.shape == (50,)

    def test_gradient_flows(self):
        """Gradients can be computed (no detach or in-place bugs)."""
        net = ContinuationNet(input_dim=2, hidden_width=16, depth=3)
        x = torch.randn(20, 2, requires_grad=False)
        out = net(x)
        loss = out.mean()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None

    def test_parameter_count_sanity(self):
        """Network has > 0 trainable parameters."""
        net = ContinuationNet(input_dim=3, hidden_width=32, depth=3)
        n_params = sum(p.numel() for p in net.parameters())
        assert n_params > 0

    def test_reproducibility_with_seed(self):
        """Same seed → identical forward pass."""
        torch.manual_seed(0)
        net1 = ContinuationNet(input_dim=2, hidden_width=16, depth=3)
        torch.manual_seed(0)
        net2 = ContinuationNet(input_dim=2, hidden_width=16, depth=3)
        x = torch.randn(10, 2)
        torch.testing.assert_close(net1(x), net2(x))


# ─────────────────────────────────────────────────────────────────────────────
# train_network
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainNetwork:
    def _make_data(self, n=200, d=1, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d)).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        return X, y

    def test_returns_loss_history_of_correct_length(self):
        """Loss history has one entry per epoch."""
        net = ContinuationNet(1, 16, 2)
        X, y = self._make_data()
        X_t = torch.tensor(X)
        y_t = torch.tensor(y)
        losses = train_network(net, X_t, y_t, n_epochs=20)
        assert len(losses) == 20

    def test_loss_decreases_on_simple_regression(self):
        """Loss should strictly decrease on a simple regression task."""
        torch.manual_seed(0)
        net = ContinuationNet(1, 32, 3)
        # Simple y = 2x + noise
        n = 1000
        X = torch.linspace(-1, 1, n).unsqueeze(1)
        y = 2.0 * X.squeeze() + 0.05 * torch.randn(n)
        losses = train_network(net, X, y, n_epochs=50, lr=1e-2)
        # Last epoch loss should be below first epoch loss
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_accepts_numpy_inputs(self):
        """train_network works with numpy arrays (not just tensors)."""
        net = ContinuationNet(2, 16, 2)
        X = np.random.randn(100, 2).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        losses = train_network(net, X, y, n_epochs=5)
        assert len(losses) == 5

    def test_all_loss_values_finite(self):
        """All loss values should be finite (no NaN or inf)."""
        net = ContinuationNet(1, 32, 3)
        X, y = self._make_data()
        X_t, y_t = torch.tensor(X), torch.tensor(y)
        losses = train_network(net, X_t, y_t, n_epochs=30)
        assert all(np.isfinite(l) for l in losses)

    def test_large_batch_size_ok(self):
        """Batch size larger than dataset → single batch per epoch."""
        net = ContinuationNet(1, 16, 2)
        X = torch.randn(50, 1)
        y = torch.randn(50)
        losses = train_network(net, X, y, n_epochs=5, batch_size=8192)
        assert len(losses) == 5

    def test_network_modified_in_place(self):
        """After training, network parameters change from their initial values."""
        torch.manual_seed(7)
        net = ContinuationNet(1, 16, 2)
        initial_params = [p.clone() for p in net.parameters()]
        X = torch.randn(200, 1)
        y = torch.randn(200)
        train_network(net, X, y, n_epochs=10)
        changed = any(
            not torch.allclose(p, p_init)
            for p, p_init in zip(net.parameters(), initial_params)
        )
        assert changed, "Network parameters did not change during training"

    def test_eval_mode_after_training(self):
        """Network is in eval mode after train_network returns."""
        net = ContinuationNet(1, 16, 2)
        X, y = torch.randn(50, 1), torch.randn(50)
        train_network(net, X, y, n_epochs=5)
        assert not net.training, "Network should be in eval mode after training"
