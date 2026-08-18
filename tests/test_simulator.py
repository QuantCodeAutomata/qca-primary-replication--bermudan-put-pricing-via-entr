"""
Tests for src/simulator.py.

Validates Black-Scholes path simulation, payoff computation, and
exercise-date index mapping.
"""

import numpy as np
import pytest
from src.simulator import (
    simulate_bs_paths,
    simulate_bs_paths_nd,
    get_exercise_date_indices,
    compute_put_payoffs,
    compute_maxcall_payoffs,
)


# ─────────────────────────────────────────────────────────────────────────────
# simulate_bs_paths
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateBSPaths:
    def test_output_shapes(self):
        """times: (n_steps+1,); paths: (n_paths, n_steps+1)."""
        times, paths = simulate_bs_paths(100.0, 0.05, 0.2, 1.0, 500, 50)
        assert times.shape == (51,)
        assert paths.shape == (500, 51)

    def test_initial_price(self):
        """All paths start at x0 at time 0."""
        x0 = 120.0
        _, paths = simulate_bs_paths(x0, 0.05, 0.2, 1.0, 1000, 20, seed=42)
        np.testing.assert_allclose(paths[:, 0], x0, rtol=1e-10)

    def test_all_positive(self):
        """GBM paths must remain strictly positive."""
        _, paths = simulate_bs_paths(100.0, 0.05, 0.3, 5.0, 2000, 100, seed=0)
        assert np.all(paths > 0)

    def test_reproducibility_with_seed(self):
        """Same seed → same paths."""
        _, p1 = simulate_bs_paths(100.0, 0.05, 0.2, 1.0, 100, 10, seed=7)
        _, p2 = simulate_bs_paths(100.0, 0.05, 0.2, 1.0, 100, 10, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_different_paths(self):
        """Different seeds → different paths."""
        _, p1 = simulate_bs_paths(100.0, 0.05, 0.2, 1.0, 100, 10, seed=1)
        _, p2 = simulate_bs_paths(100.0, 0.05, 0.2, 1.0, 100, 10, seed=2)
        assert not np.allclose(p1, p2)

    def test_time_grid(self):
        """Time grid spans [0, T] uniformly."""
        T = 1.5
        n_steps = 30
        times, _ = simulate_bs_paths(100.0, 0.05, 0.2, T, 50, n_steps)
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(T)
        np.testing.assert_allclose(np.diff(times), T / n_steps, rtol=1e-12)

    def test_risk_neutral_mean(self):
        """E[X_T] ≈ x0 * exp((r - delta)*T) under risk-neutral measure."""
        x0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
        n_paths = 50_000
        _, paths = simulate_bs_paths(x0, r, sigma, T, n_paths, 1, seed=99)
        expected_mean = x0 * np.exp(r * T)
        actual_mean = np.mean(paths[:, -1])
        # 3-sigma interval for sample mean
        std_mean = x0 * np.exp(r * T) * sigma * np.sqrt(T) / np.sqrt(n_paths)
        assert abs(actual_mean - expected_mean) < 4 * std_mean, \
            f"E[X_T]={actual_mean:.2f} vs expected {expected_mean:.2f}"

    def test_no_dividend_vs_with_dividend(self):
        """Dividend reduces expected terminal price by exp(-delta*T)."""
        x0, r, sigma, T, delta = 100.0, 0.05, 0.2, 1.0, 0.03
        n = 50_000
        _, paths_no = simulate_bs_paths(x0, r, sigma, T, n, 1, seed=10, delta=0.0)
        _, paths_div = simulate_bs_paths(x0, r, sigma, T, n, 1, seed=10, delta=delta)
        ratio = np.mean(paths_div[:, -1]) / np.mean(paths_no[:, -1])
        np.testing.assert_allclose(ratio, np.exp(-delta * T), rtol=0.05)

    def test_single_step(self):
        """Single-step simulation still returns correct shapes."""
        times, paths = simulate_bs_paths(50.0, 0.1, 0.25, 0.5, 10, 1, seed=0)
        assert times.shape == (2,)
        assert paths.shape == (10, 2)


# ─────────────────────────────────────────────────────────────────────────────
# simulate_bs_paths_nd
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateBSPathsND:
    def test_output_shapes(self):
        """times: (n_steps+1,); paths: (n_paths, n_steps+1, d)."""
        d = 5
        times, paths = simulate_bs_paths_nd(100.0, 0.05, 0.2, 3.0, 200, 30, d=d)
        assert times.shape == (31,)
        assert paths.shape == (200, 31, d)

    def test_initial_price_all_assets(self):
        """All d assets start at x0."""
        x0, d = 80.0, 4
        _, paths = simulate_bs_paths_nd(x0, 0.05, 0.2, 1.0, 100, 10, d=d, seed=42)
        np.testing.assert_allclose(paths[:, 0, :], x0, rtol=1e-10)

    def test_all_positive(self):
        """All asset prices remain positive."""
        _, paths = simulate_bs_paths_nd(100.0, 0.05, 0.3, 3.0, 500, 50, d=10, seed=1)
        assert np.all(paths > 0)

    def test_independence_across_assets(self):
        """Asset paths should be correlated only through shared Brownian structure."""
        d = 2
        n_paths = 10_000
        _, paths = simulate_bs_paths_nd(100.0, 0.05, 0.2, 1.0, n_paths, 1, d=d, seed=5)
        # Log-returns at maturity for each asset
        log_returns_0 = np.log(paths[:, 1, 0] / paths[:, 0, 0])
        log_returns_1 = np.log(paths[:, 1, 1] / paths[:, 0, 1])
        corr = np.corrcoef(log_returns_0, log_returns_1)[0, 1]
        # With independent Brownians, correlation should be near 0
        assert abs(corr) < 0.05, f"Unexpected correlation {corr:.4f}"

    def test_reproducibility(self):
        """Same seed produces identical multi-asset paths."""
        kw = dict(x0=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=50, n_steps=5, d=3)
        _, p1 = simulate_bs_paths_nd(**kw, seed=42)
        _, p2 = simulate_bs_paths_nd(**kw, seed=42)
        np.testing.assert_array_equal(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# get_exercise_date_indices
# ─────────────────────────────────────────────────────────────────────────────

class TestGetExerciseDateIndices:
    def test_exact_match(self):
        """Dates that fall exactly on grid points → correct indices."""
        times = np.linspace(0.0, 1.0, 11)  # 0, 0.1, ..., 1.0
        indices = get_exercise_date_indices(times, [0.0, 0.5, 1.0])
        assert indices == [0, 5, 10]

    def test_nearest_match(self):
        """Non-exact dates snap to nearest grid point."""
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        indices = get_exercise_date_indices(times, [0.26, 0.74])
        assert indices == [1, 3]

    def test_length(self):
        """Returns same number of indices as exercise_dates."""
        times = np.linspace(0, 3, 100)
        ex_dates = [1.0, 2.0, 3.0]
        indices = get_exercise_date_indices(times, ex_dates)
        assert len(indices) == 3

    def test_maturity_last_index(self):
        """Maturity T maps to the last index of times."""
        times = np.linspace(0, 1, 5)  # 0.0, 0.25, 0.5, 0.75, 1.0
        idx = get_exercise_date_indices(times, [1.0])
        assert idx == [4]


# ─────────────────────────────────────────────────────────────────────────────
# compute_put_payoffs
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePutPayoffs:
    def setup_method(self):
        """Small deterministic path set for testing."""
        self.x0, self.K, self.r = 100.0, 100.0, 0.05
        self.n_paths = 4
        # times: 0, 0.5, 1.0
        self.times = np.array([0.0, 0.5, 1.0])
        # Paths: above and below K at each date
        self.paths = np.array([
            [100.0, 90.0, 80.0],   # ITM at t=0.5 and t=1
            [100.0, 110.0, 120.0], # OTM
            [100.0, 100.0, 100.0], # ATM
            [100.0, 95.0, 105.0],  # ITM then OTM
        ])

    def test_shape(self):
        """Output shape: (n_paths, len(exercise_indices))."""
        ex_idx = [0, 1, 2]
        payoffs = compute_put_payoffs(self.paths, self.times, self.K, self.r, ex_idx)
        assert payoffs.shape == (self.n_paths, 3)

    def test_discounting(self):
        """Payoffs are discounted by exp(-r*t)."""
        ex_idx = [1]  # t=0.5
        payoffs = compute_put_payoffs(self.paths, self.times, self.K, self.r, ex_idx)
        expected = np.exp(-self.r * 0.5) * np.maximum(self.K - self.paths[:, 1], 0.0)
        np.testing.assert_allclose(payoffs[:, 0], expected, rtol=1e-10)

    def test_nonnegative(self):
        """All put payoffs are ≥ 0."""
        ex_idx = [0, 1, 2]
        payoffs = compute_put_payoffs(self.paths, self.times, self.K, self.r, ex_idx)
        assert np.all(payoffs >= 0)

    def test_otm_zero(self):
        """Out-of-the-money paths yield zero payoff."""
        ex_idx = [2]  # path 1 ends at 120 > K=100
        payoffs = compute_put_payoffs(self.paths, self.times, self.K, self.r, ex_idx)
        assert payoffs[1, 0] == pytest.approx(0.0)

    def test_at_time_zero_no_discount(self):
        """At t=0, discount factor = 1, payoff = max(K-X,0)."""
        ex_idx = [0]
        payoffs = compute_put_payoffs(self.paths, self.times, self.K, self.r, ex_idx)
        expected = np.maximum(self.K - self.paths[:, 0], 0.0)
        np.testing.assert_allclose(payoffs[:, 0], expected, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# compute_maxcall_payoffs
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMaxcallPayoffs:
    def setup_method(self):
        self.K, self.r = 100.0, 0.05
        self.times = np.array([0.0, 0.5, 1.0])
        # paths_nd: (n_paths=3, n_times=3, d=2)
        self.paths_nd = np.array([
            [[100.0, 90.0], [110.0, 80.0], [95.0, 120.0]],
            [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]],
            [[100.0, 50.0], [60.0, 50.0], [70.0, 55.0]],
        ], dtype=float)

    def test_shape(self):
        """Output: (n_paths, n_exercise_dates)."""
        ex_idx = [0, 1, 2]
        payoffs = compute_maxcall_payoffs(self.paths_nd, self.times, self.K, self.r, ex_idx)
        assert payoffs.shape == (3, 3)

    def test_maxcall_value(self):
        """Payoff = exp(-r*t) * max(max_i X_i - K, 0)."""
        ex_idx = [1]  # t=0.5
        payoffs = compute_maxcall_payoffs(self.paths_nd, self.times, self.K, self.r, ex_idx)
        # Path 0: max(110, 80)=110, payoff = exp(-0.025)*(110-100)
        expected_0 = np.exp(-self.r * 0.5) * max(110.0 - self.K, 0)
        np.testing.assert_allclose(payoffs[0, 0], expected_0, rtol=1e-10)

    def test_nonnegative(self):
        """All max-call payoffs are ≥ 0."""
        ex_idx = [0, 1, 2]
        payoffs = compute_maxcall_payoffs(self.paths_nd, self.times, self.K, self.r, ex_idx)
        assert np.all(payoffs >= 0)

    def test_otm_gives_zero(self):
        """All assets below K → payoff = 0."""
        ex_idx = [2]  # Path 2: max(70, 55)=70 < K=100
        payoffs = compute_maxcall_payoffs(self.paths_nd, self.times, self.K, self.r, ex_idx)
        assert payoffs[2, 0] == pytest.approx(0.0)
