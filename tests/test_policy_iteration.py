"""
Tests for src/policy_iteration.py.

Validates the PolicyIterationSolver interface, initialisation,
iteration mechanics, and qualitative convergence properties.
"""

import numpy as np
import pytest
from src.policy_iteration import PolicyIterationSolver


def make_bermudan_put_data(
    n_paths: int = 256,
    N: int = 3,
    x0: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.1,
    K: float = 100.0,
    T: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Generate synthetic Bermudan put data with exercise times.

    Returns
    -------
    paths : np.ndarray, shape (n_paths, N+2, 1)
    payoffs : np.ndarray, shape (n_paths, N+2)
    exercise_times : list[float]  length N+2
    """
    rng = np.random.default_rng(seed)
    exercise_times = [i * T / (N + 1) for i in range(N + 1)] + [T]
    dt = exercise_times[1]

    paths_1d = np.zeros((n_paths, N + 2))
    paths_1d[:, 0] = x0
    for i in range(1, N + 2):
        z = rng.standard_normal(n_paths)
        paths_1d[:, i] = paths_1d[:, i - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    paths = paths_1d[:, :, np.newaxis]
    payoffs = np.zeros((n_paths, N + 2))
    for i, t in enumerate(exercise_times):
        payoffs[:, i] = np.exp(-r * t) * np.maximum(K - paths_1d[:, i], 0.0)

    return paths, payoffs, exercise_times


# ─────────────────────────────────────────────────────────────────────────────
# Constructor and interface
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicySolverInterface:
    def test_constructor_stores_params(self):
        """Constructor stores all parameters correctly."""
        s = PolicyIterationSolver(
            lam=0.05, n_exercise_dates=4, state_dim=2,
            hidden_width=48, depth=4, n_epochs=20, lr=2e-3,
        )
        assert s.lam == 0.05
        assert s.n_exercise_dates == 4
        assert s.state_dim == 2
        assert s.hidden_width == 48
        assert s.depth == 4
        assert s.n_epochs == 20
        assert s.lr == pytest.approx(2e-3)

    def test_empty_sequence_before_fit(self):
        """v0_sequence is empty before calling fit."""
        s = PolicyIterationSolver(lam=0.1, n_exercise_dates=3)
        assert s.v0_sequence == []

    def test_fit_returns_list_of_floats(self):
        """fit() returns a list of floats."""
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=64, N=3, seed=0)
        s = PolicyIterationSolver(lam=0.1, n_exercise_dates=3, n_epochs=5)
        result = s.fit(paths, payoffs, ex_times, max_iter=3)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_sequence_length_matches_max_iter_plus_one(self):
        """fit() returns max_iter + 1 values (initialization + max_iter iterations)."""
        max_iter = 4
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=64, N=3, seed=1)
        s = PolicyIterationSolver(lam=0.1, n_exercise_dates=3, n_epochs=5)
        result = s.fit(paths, payoffs, ex_times, max_iter=max_iter)
        assert len(result) == max_iter + 1

    def test_time0_price_sequence_accessor(self):
        """time0_price_sequence() returns same list as fit()."""
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=64, N=3, seed=2)
        s = PolicyIterationSolver(lam=0.1, n_exercise_dates=3, n_epochs=5)
        fit_result = s.fit(paths, payoffs, ex_times, max_iter=2)
        assert s.time0_price_sequence() == fit_result

    def test_v0_init_positive(self):
        """The initialisation value v_0^{λ,0} is strictly positive (European put price)."""
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=256, N=3, seed=3)
        s = PolicyIterationSolver(lam=0.1, n_exercise_dates=3, n_epochs=5)
        result = s.fit(paths, payoffs, ex_times, max_iter=2)
        assert result[0] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative properties (using fast, smaller networks for speed)
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicySolverQuality:
    @pytest.fixture(scope="class")
    def results_three_lam(self):
        """Run policy iteration for three λ values."""
        paths, payoffs, ex_times = make_bermudan_put_data(
            n_paths=512, N=3, seed=42
        )
        results = {}
        for lam in [0.1, 0.01, 0.001]:
            s = PolicyIterationSolver(
                lam=lam, n_exercise_dates=3, state_dim=1,
                hidden_width=32, depth=3, n_epochs=200, lr=1e-3,
            )
            seq = s.fit(paths, payoffs, ex_times, max_iter=5,
                        use_bs_init=True)
            results[lam] = seq
        return results

    def test_init_close_to_european_put(self, results_three_lam):
        """v_0^{λ,0} should be close to the European put price ≈ 1.928."""
        for lam, seq in results_three_lam.items():
            # European put price for x0=100, r=0.05, sigma=0.1, K=100, T=1: ≈1.928
            assert 0.5 < seq[0] < 5.0, \
                f"Initialisation {seq[0]:.4f} too far from European put for λ={lam}"

    def test_price_in_reasonable_range(self, results_three_lam):
        """
        Final regularised Bermudan price should be positive and bounded.

        Note: for entropy-regularised options, the price can be below the
        European put for large λ, so we only verify the value stays in a
        financially meaningful range [0.1, 20] (below K=100 is checked
        separately).
        """
        for lam, seq in results_three_lam.items():
            assert seq[-1] > 0.1, \
                f"Final price {seq[-1]:.4f} implausibly low for λ={lam}"
            assert seq[-1] < 20.0, \
                f"Final price {seq[-1]:.4f} implausibly high for λ={lam}"

    def test_all_prices_finite(self, results_three_lam):
        """Every iteration price should be finite."""
        for lam, seq in results_three_lam.items():
            assert all(np.isfinite(v) for v in seq), \
                f"Non-finite price in sequence for λ={lam}: {seq}"

    def test_all_prices_positive(self, results_three_lam):
        """Every iteration price should be strictly positive."""
        for lam, seq in results_three_lam.items():
            assert all(v > 0 for v in seq), \
                f"Non-positive price in sequence for λ={lam}: {seq}"

    def test_price_below_strike(self, results_three_lam):
        """All prices should be below K=100."""
        for lam, seq in results_three_lam.items():
            assert all(v < 100.0 for v in seq), \
                f"Price exceeds strike for λ={lam}: {seq}"

    def test_larger_lam_lower_or_equal_price(self, results_three_lam):
        """
        Larger λ → lower or equal entropy-regularised price (prices increase as λ→0).
        Compare final iteration values.
        """
        v_01 = results_three_lam[0.1][-1]
        v_001 = results_three_lam[0.01][-1]
        v_0001 = results_three_lam[0.001][-1]
        # With tolerance for Monte Carlo noise
        tol = 0.5
        assert v_01 <= v_001 + tol, f"Expected v(λ=0.1) ≤ v(λ=0.01): {v_01:.3f} vs {v_001:.3f}"
        assert v_001 <= v_0001 + tol, f"Expected v(λ=0.01) ≤ v(λ=0.001): {v_001:.3f} vs {v_0001:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicySolverEdgeCases:
    def test_single_exercise_date(self):
        """N=1 single pre-maturity exercise date runs without error."""
        N = 1
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=64, N=N, seed=5)
        s = PolicyIterationSolver(lam=0.1, n_exercise_dates=N, n_epochs=5)
        seq = s.fit(paths, payoffs, ex_times, max_iter=2)
        assert len(seq) == 3
        assert all(np.isfinite(v) for v in seq)

    def test_zero_iterations_returns_init(self):
        """max_iter=0 returns only the initialization value."""
        N = 3
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=64, N=N, seed=6)
        s = PolicyIterationSolver(lam=0.1, n_exercise_dates=N, n_epochs=5)
        seq = s.fit(paths, payoffs, ex_times, max_iter=0)
        assert len(seq) == 1

    def test_non_bs_initialisation(self):
        """MC-regression initialisation (use_bs_init=False) runs without error."""
        N = 3
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=128, N=N, seed=7)
        s = PolicyIterationSolver(
            lam=0.1, n_exercise_dates=N, n_epochs=5
        )
        seq = s.fit(paths, payoffs, ex_times, max_iter=2, use_bs_init=False)
        assert len(seq) == 3
        assert all(np.isfinite(v) for v in seq)

    def test_small_lambda_no_nan(self):
        """λ=0.001 (small regularisation) produces finite prices."""
        N = 3
        paths, payoffs, ex_times = make_bermudan_put_data(n_paths=64, N=N, seed=8)
        s = PolicyIterationSolver(lam=0.001, n_exercise_dates=N, n_epochs=5)
        seq = s.fit(paths, payoffs, ex_times, max_iter=3)
        assert all(np.isfinite(v) for v in seq)
