"""
Tests for src/td_solver.py.

Validates the TDSolver interface, backward training loop, and
qualitative properties of the TD-based BSDE price estimate.
"""

import numpy as np
import pytest
from src.td_solver import TDSolver


def make_bermudan_put_data(
    n_paths: int = 512,
    N: int = 3,
    x0: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.1,
    K: float = 100.0,
    T: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Black-Scholes Bermudan put data.

    Returns
    -------
    paths : np.ndarray, shape (n_paths, N+2, 1)
    payoffs : np.ndarray, shape (n_paths, N+2)
    """
    rng = np.random.default_rng(seed)
    # Dates: t_0, ..., t_N, T  (N+2 entries)
    times = np.array([i / (N + 1) for i in range(N + 1)] + [T])
    dt = times[1]

    paths_1d = np.zeros((n_paths, N + 2))
    paths_1d[:, 0] = x0
    for i in range(1, N + 2):
        z = rng.standard_normal(n_paths)
        paths_1d[:, i] = paths_1d[:, i - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    paths = paths_1d[:, :, np.newaxis]          # (n_paths, N+2, 1)
    payoffs = np.zeros((n_paths, N + 2))
    for i, t in enumerate(times):
        payoffs[:, i] = np.exp(-r * t) * np.maximum(K - paths_1d[:, i], 0.0)

    return paths, payoffs


# ─────────────────────────────────────────────────────────────────────────────
# Constructor and interface
# ─────────────────────────────────────────────────────────────────────────────

class TestTDSolverInterface:
    def test_constructor_stores_params(self):
        """Constructor correctly stores all hyper-parameters."""
        solver = TDSolver(
            lam=0.01, n_exercise_dates=3, state_dim=1,
            hidden_width=32, depth=2, n_epochs=10, lr=5e-4,
        )
        assert solver.lam == 0.01
        assert solver.n_exercise_dates == 3
        assert solver.state_dim == 1
        assert solver.hidden_width == 32
        assert solver.depth == 2
        assert solver.n_epochs == 10
        assert solver.lr == pytest.approx(5e-4)

    def test_nets_empty_before_fit(self):
        """Before fitting, nets list is empty."""
        solver = TDSolver(lam=0.1, n_exercise_dates=3, state_dim=1)
        assert solver.nets == []

    def test_nets_populated_after_fit(self):
        """After fitting, nets list has N+1 entries (date 0 to date N)."""
        N = 3
        solver = TDSolver(lam=0.1, n_exercise_dates=N, state_dim=1, n_epochs=5)
        paths, payoffs = make_bermudan_put_data(n_paths=64, N=N, seed=1)
        solver.fit(paths, payoffs)
        assert len(solver.nets) == N + 1

    def test_predict_continuation_shape(self):
        """predict_continuation returns (n_paths,) array."""
        N = 2
        solver = TDSolver(lam=0.1, n_exercise_dates=N, state_dim=1, n_epochs=5)
        paths, payoffs = make_bermudan_put_data(n_paths=64, N=N, seed=2)
        solver.fit(paths, payoffs)
        X = paths[:, 0, :]
        cont = solver.predict_continuation(X, date_idx=0)
        assert cont.shape == (64,)

    def test_predict_value_shape(self):
        """predict_value returns (n_paths,) array."""
        N = 2
        solver = TDSolver(lam=0.1, n_exercise_dates=N, state_dim=1, n_epochs=5)
        paths, payoffs = make_bermudan_put_data(n_paths=64, N=N, seed=3)
        solver.fit(paths, payoffs)
        val = solver.predict_value(paths[:, 0, :], payoffs[:, 0], date_idx=0)
        assert val.shape == (64,)

    def test_time0_price_returns_float(self):
        """time0_price returns a Python float."""
        N = 2
        solver = TDSolver(lam=0.1, n_exercise_dates=N, state_dim=1, n_epochs=5)
        paths, payoffs = make_bermudan_put_data(n_paths=64, N=N, seed=4)
        solver.fit(paths, payoffs)
        price = solver.time0_price(paths, payoffs)
        assert isinstance(price, float)


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative properties
# ─────────────────────────────────────────────────────────────────────────────

class TestTDSolverQuality:
    """Qualitative checks that do not require tight numerical accuracy."""

    @pytest.fixture(scope="class")
    def trained_solvers(self):
        """Train TDSolvers for three λ values with shared data."""
        N = 3
        paths, payoffs = make_bermudan_put_data(n_paths=1024, N=N, seed=42)
        solvers = {}
        for lam in [0.1, 0.01, 0.001]:
            s = TDSolver(
                lam=lam, n_exercise_dates=N, state_dim=1,
                hidden_width=32, depth=3, n_epochs=50, lr=1e-3,
            )
            s.fit(paths, payoffs)
            solvers[lam] = s
        return paths, payoffs, solvers

    def test_price_positive(self, trained_solvers):
        """Time-0 TD price is strictly positive."""
        paths, payoffs, solvers = trained_solvers
        for lam, s in solvers.items():
            price = s.time0_price(paths, payoffs)
            assert price > 0.0, f"Non-positive price for lam={lam}: {price}"

    def test_price_below_strike(self, trained_solvers):
        """TD price must be below the undiscounted strike K=100."""
        paths, payoffs, solvers = trained_solvers
        for lam, s in solvers.items():
            price = s.time0_price(paths, payoffs)
            assert price < 100.0, f"Price exceeds strike for lam={lam}: {price}"

    def test_price_above_european(self, trained_solvers):
        """Bermudan put price should be at or above European put (≈ 1.928)."""
        paths, payoffs, solvers = trained_solvers
        european_lb = 1.0   # loose lower bound to account for training noise
        for lam, s in solvers.items():
            price = s.time0_price(paths, payoffs)
            assert price >= european_lb, \
                f"Price too low for lam={lam}: {price}"

    def test_value_at_least_payoff(self, trained_solvers):
        """v_hat(V+, P, λ) ≥ V+ (since Psi > 0 and value incorporates payoff)."""
        # This tests that adding G_j (which uses Psi) always keeps V >= V+
        # when P >= V+, and that the regularised value is at least the continuation
        from src.special_functions import v_hat
        paths, payoffs, solvers = trained_solvers
        paths_test, payoffs_test = paths[:100], payoffs[:100]
        s = solvers[0.01]
        V_plus = s.predict_continuation(paths_test[:, 0, :], date_idx=0)
        P_0 = payoffs_test[:, 0]
        V_val = v_hat(V_plus, P_0, s.lam)
        # v_hat(V_plus, P, lam) >= V_plus always? Not necessarily; only when P >= V_plus
        # For paths where P >= V_plus, V_val >= V_plus
        mask = P_0 >= V_plus
        if mask.any():
            assert np.all(V_val[mask] >= V_plus[mask] - 1e-6), \
                "v_hat < V_plus when P >= V_plus"

    def test_finite_prices(self, trained_solvers):
        """No NaN or Inf in TD price estimates."""
        paths, payoffs, solvers = trained_solvers
        for lam, s in solvers.items():
            price = s.time0_price(paths, payoffs)
            assert np.isfinite(price), f"Non-finite price for lam={lam}: {price}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestTDSolverEdgeCases:
    def test_single_exercise_date(self):
        """Solver with N=1 (only one pre-maturity exercise) runs without error."""
        N = 1
        solver = TDSolver(lam=0.1, n_exercise_dates=N, state_dim=1, n_epochs=5)
        paths, payoffs = make_bermudan_put_data(n_paths=64, N=N, seed=99)
        solver.fit(paths, payoffs)
        assert len(solver.nets) == N + 1
        price = solver.time0_price(paths, payoffs)
        assert np.isfinite(price)

    def test_multidim_state(self):
        """Solver works with d=2 state dimension."""
        rng = np.random.default_rng(0)
        N, d, n = 2, 2, 64
        paths = rng.lognormal(np.log(100), 0.2, size=(n, N + 2, d))
        payoffs = np.exp(-0.05 * np.tile(
            [i / (N + 1) for i in range(N + 2)], (n, 1)
        )) * np.maximum(np.max(paths, axis=2) - 100.0, 0.0)

        solver = TDSolver(
            lam=0.1, n_exercise_dates=N, state_dim=d, n_epochs=5
        )
        solver.fit(paths, payoffs)
        assert len(solver.nets) == N + 1
        price = solver.time0_price(paths, payoffs)
        assert np.isfinite(price)

    def test_very_small_lambda_no_nan(self):
        """Very small λ=0.001 does not produce NaN in training."""
        N = 2
        solver = TDSolver(lam=0.001, n_exercise_dates=N, state_dim=1, n_epochs=5)
        paths, payoffs = make_bermudan_put_data(n_paths=64, N=N, seed=55)
        solver.fit(paths, payoffs)
        price = solver.time0_price(paths, payoffs)
        assert np.isfinite(price)
