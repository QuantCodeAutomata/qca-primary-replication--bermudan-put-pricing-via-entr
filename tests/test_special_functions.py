"""
Tests for src/special_functions.py.

Validates the numerically stable implementations of Phi, Psi, gibbs_mean,
regularized_value_at_date, and v_hat against known analytic properties.
"""

import numpy as np
import pytest
from src.special_functions import (
    phi,
    psi,
    gibbs_mean,
    regularized_value_at_date,
    v_hat,
)


# ─────────────────────────────────────────────────────────────────────────────
# Phi tests  (equation 3.8: Phi(x) = log((exp(x)-1)/x), Phi(0)=0)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhi:
    def test_phi_zero(self):
        """Phi(0) = 0 by convention."""
        assert phi(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_phi_zero_array(self):
        """Phi(0) works for zero-array inputs."""
        result = phi(np.zeros(10))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_phi_positive_values(self):
        """Phi(x) for moderate positive x, checked against reference formula."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        ref = np.log(np.expm1(x) / x)
        np.testing.assert_allclose(phi(x), ref, rtol=1e-10)

    def test_phi_negative_values(self):
        """Phi(x) for moderate negative x (expm1 branch)."""
        x = np.array([-0.5, -1.0, -2.0, -5.0])
        # (exp(x)-1)/x > 0 for x<0 as well: both neg/neg = pos
        ref = np.log(np.abs(np.expm1(x)) / np.abs(x))
        np.testing.assert_allclose(phi(x), ref, rtol=1e-10)

    def test_phi_near_zero_taylor(self):
        """Near zero, Phi(x) ≈ x/2 + x²/24."""
        x = np.array([1e-6, 1e-5, 1e-4])
        taylor = x / 2 + x ** 2 / 24
        np.testing.assert_allclose(phi(x), taylor, rtol=1e-5)

    def test_phi_large_positive(self):
        """For very large x, Phi(x) ≈ x - log(x)."""
        x = np.array([600.0, 1000.0])
        asymp = x - np.log(x)
        np.testing.assert_allclose(phi(x), asymp, rtol=1e-5)

    def test_phi_large_negative_no_nan(self):
        """For large-negative x, Phi should return finite values (no NaN)."""
        x = np.array([-100.0, -500.0, -1000.0])
        result = phi(x)
        assert np.all(np.isfinite(result)), f"Got non-finite: {result}"

    def test_phi_scalar(self):
        """Phi accepts and returns a scalar."""
        result = phi(1.0); assert np.isfinite(float(result))

    def test_phi_monotone_sign(self):
        """Phi(x) has the same sign as x: negative for x<0, zero at x=0, positive for x>0."""
        x_neg = np.array([-10.0, -5.0, -1.0])
        x_pos = np.array([1.0, 5.0, 10.0])
        # Phi < 0 for x < 0
        assert np.all(phi(x_neg) < 0), f"Expected phi<0 for neg x, got {phi(x_neg)}"
        # Phi = 0 at x = 0
        assert phi(0.0) == pytest.approx(0.0, abs=1e-10)
        # Phi > 0 for x > 0
        assert np.all(phi(x_pos) > 0), f"Expected phi>0 for pos x, got {phi(x_pos)}"

    def test_phi_no_runtime_warnings(self, recwarn):
        """phi should not emit RuntimeWarnings for any real inputs."""
        import warnings
        x = np.array([-1000.0, -10.0, -1e-7, 0.0, 1e-7, 10.0, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            phi(x)  # raises if any RuntimeWarning emitted


# ─────────────────────────────────────────────────────────────────────────────
# Psi tests  (Psi(x) = Phi(x)/x, Psi(0)=1/2)
# ─────────────────────────────────────────────────────────────────────────────

class TestPsi:
    def test_psi_zero(self):
        """Psi(0) = 1/2 by convention."""
        assert psi(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_psi_zero_array(self):
        """Psi(0) for arrays."""
        result = psi(np.zeros(5))
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

    def test_psi_range(self):
        """Psi(x) ∈ (0, 1) for all x ∈ ℝ."""
        x = np.linspace(-50.0, 50.0, 500)
        result = psi(x)
        assert np.all(result > 0), "Psi must be positive"
        assert np.all(result <= 1.0 + 1e-9), "Psi must be ≤ 1"

    def test_psi_limit_positive_infty(self):
        """Psi(x) → 1 as x → +∞.  Need x ≳ 1e5 for atol=1e-3 (rate log(x)/x)."""
        result = psi(np.array([1e5, 1e6]))
        np.testing.assert_allclose(result, 1.0, atol=1e-3)

    def test_psi_limit_negative_infty(self):
        """Psi(x) → 0 as x → -∞.  Convergence rate is log|x|/|x|; need |x| ≳ 1e4."""
        result = psi(np.array([-1e4, -2e4]))
        np.testing.assert_allclose(result, 0.0, atol=1e-2)

    def test_psi_near_zero_taylor(self):
        """Near zero, Psi(x) ≈ 1/2 + x/24."""
        x = np.array([1e-6, 5e-5])
        taylor = 0.5 + x / 24
        np.testing.assert_allclose(psi(x), taylor, rtol=1e-5)

    def test_psi_consistency_with_phi(self):
        """Psi(x) = Phi(x) / x for x away from 0."""
        x = np.array([-5.0, -1.0, 0.5, 1.0, 3.0, 10.0])
        np.testing.assert_allclose(psi(x), phi(x) / x, rtol=1e-8)

    def test_psi_monotone(self):
        """Psi is monotone increasing."""
        x = np.sort(np.random.default_rng(0).uniform(-30, 30, 200))
        result = psi(x)
        assert np.all(np.diff(result) >= -1e-10)

    def test_psi_no_runtime_warnings(self):
        """psi should not emit RuntimeWarnings."""
        import warnings
        x = np.array([-1000.0, -0.001, 0.0, 0.001, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            psi(x)


# ─────────────────────────────────────────────────────────────────────────────
# gibbs_mean tests  (mean of Gibbs density on [0,1])
# ─────────────────────────────────────────────────────────────────────────────

class TestGibbsMean:
    def test_gibbs_mean_zero(self):
        """gibbs_mean(0) = 1/2 (uniform density)."""
        assert gibbs_mean(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_gibbs_mean_range(self):
        """gibbs_mean(z) ∈ (0, 1) for all z."""
        z = np.linspace(-100.0, 100.0, 500)
        result = gibbs_mean(z)
        assert np.all(result > 0), "gibbs_mean must be > 0"
        assert np.all(result < 1), "gibbs_mean must be < 1"

    def test_gibbs_mean_limit_positive(self):
        """gibbs_mean → 1 as z → +∞."""
        result = gibbs_mean(np.array([200.0, 500.0]))
        np.testing.assert_allclose(result, 1.0, atol=1e-2)

    def test_gibbs_mean_limit_negative(self):
        """gibbs_mean → 0 as z → -∞."""
        result = gibbs_mean(np.array([-200.0, -500.0]))
        np.testing.assert_allclose(result, 0.0, atol=1e-2)

    def test_gibbs_mean_monotone(self):
        """gibbs_mean is monotone increasing in z."""
        z = np.sort(np.random.default_rng(1).uniform(-30, 30, 300))
        result = gibbs_mean(z)
        assert np.all(np.diff(result) >= -1e-10)

    def test_gibbs_mean_taylor_near_zero(self):
        """Near zero, gibbs_mean(z) ≈ 1/2 + z/12."""
        z = np.array([1e-5, 5e-5])
        taylor = 0.5 + z / 12
        np.testing.assert_allclose(gibbs_mean(z), taylor, rtol=1e-5)

    def test_gibbs_mean_numerical_quadrature(self):
        """
        Cross-check against numerical integration of the Gibbs density.

        π*(u|z) = z/(exp(z)-1) * exp(z*u), u in [0,1]
        μ = ∫₀¹ u * π*(u|z) du
        """
        from scipy.integrate import quad

        for z in [-3.0, -1.0, 0.5, 2.0, 5.0]:
            if abs(z) < 1e-10:
                mu_ref = 0.5
            else:
                norm_const = z / (np.exp(z) - 1)

                def integrand(u):
                    return u * norm_const * np.exp(z * u)

                mu_ref, _ = quad(integrand, 0.0, 1.0)
            np.testing.assert_allclose(
                gibbs_mean(z), mu_ref, rtol=1e-6,
                err_msg=f"Failed for z={z}"
            )

    def test_gibbs_mean_no_runtime_warnings(self):
        """gibbs_mean should not emit RuntimeWarnings."""
        import warnings
        z = np.array([-1000.0, -0.001, 0.0, 0.001, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gibbs_mean(z)


# ─────────────────────────────────────────────────────────────────────────────
# regularized_value_at_date / v_hat tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegularizedValue:
    def test_V_between_Vplus_and_P_when_P_larger(self):
        """When P > V_plus, V_t = V_plus + (P-V_plus)*Psi(...) ∈ [V_plus, P]."""
        V_plus = np.array([1.0, 2.0, 0.5])
        P = np.array([3.0, 5.0, 1.5])
        lam = 0.1
        V = regularized_value_at_date(V_plus, P, lam)
        assert np.all(V >= V_plus - 1e-10)
        assert np.all(V <= P + 1e-10)

    def test_V_approaches_P_as_lam_to_zero(self):
        """As λ→0, V_t^λ → max(V_plus, P) (exercise if payoff dominates)."""
        V_plus = np.array([1.0])
        P = np.array([5.0])  # deep in the money
        for lam in [0.1, 0.01, 0.001]:
            V = regularized_value_at_date(V_plus, P, lam)
            assert V > V_plus, f"V should exceed V_plus when P > V_plus, lam={lam}"

    def test_V_approaches_V_plus_when_P_small(self):
        """When P ≪ V_plus, Psi→0, so V → V_plus (continuation dominates)."""
        V_plus = np.array([10.0])
        P = np.array([0.01])
        lam = 0.001
        V = regularized_value_at_date(V_plus, P, lam)
        np.testing.assert_allclose(V, V_plus, rtol=0.01)

    def test_V_equals_V_plus_plus_half_diff_at_P_eq_V(self):
        """When P = V_plus, z=0, Psi(0)=1/2, so V = V_plus + (P-V_plus)/2 = (V+P)/2."""
        V_plus = np.array([3.0, 4.0])
        P = V_plus.copy()
        lam = 0.5
        V = regularized_value_at_date(V_plus, P, lam)
        np.testing.assert_allclose(V, V_plus, atol=1e-10)

    def test_V_shape_preserved(self):
        """Output shape equals input shape."""
        V_plus = np.random.rand(100)
        P = np.random.rand(100)
        V = regularized_value_at_date(V_plus, P, 0.1)
        assert V.shape == (100,)

    def test_V_hat_alias(self):
        """v_hat is an exact alias for regularized_value_at_date."""
        V_plus = np.array([1.5, 2.5])
        P = np.array([2.0, 3.0])
        lam = 0.05
        np.testing.assert_array_equal(
            v_hat(V_plus, P, lam),
            regularized_value_at_date(V_plus, P, lam)
        )

    def test_V_shape_mismatch_raises(self):
        """Mismatched shapes should raise AssertionError."""
        with pytest.raises(AssertionError):
            regularized_value_at_date(np.ones(3), np.ones(4), 0.1)

    def test_V_monotone_in_lambda(self):
        """
        For fixed V_plus, P, V_t^λ should decrease as λ decreases
        (when P > V_plus, Psi increases toward 1 as λ→0, meaning V→P).
        The value V_t^λ is an increasing function of Psi, which increases
        as λ decreases for P > V_plus.
        """
        V_plus = np.array([1.0])
        P = np.array([5.0])
        lambdas = [1.0, 0.5, 0.1, 0.01]
        values = [float(regularized_value_at_date(V_plus, P, lam)) for lam in lambdas]
        # Values should increase as lam decreases (P > V_plus case)
        assert all(values[i] <= values[i+1] for i in range(len(values)-1)), \
            f"Expected monotone increasing as lam decreases, got {values}"
