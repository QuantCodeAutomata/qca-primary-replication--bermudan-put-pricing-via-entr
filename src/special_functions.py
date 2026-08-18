"""
Numerically stable implementations of the special functions used in the
entropy-regularised Bermudan option pricing framework.

All functions are fully vectorised via ``np.where`` so that no branch
evaluates to NaN/Inf on inputs that belong to a different branch.

Reference
---------
Equations (3.8) and surrounding text in
"An Entropy Regularized BSDE Approach to Bermudan Options and Games".
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NEAR_ZERO_THRESHOLD: float = 1e-3   # |x| < this → use Taylor series
_LARGE_POS_THRESHOLD: float = 500.0  # x > this  → use large-x asymptotic


def phi(x: np.ndarray) -> np.ndarray:
    """
    Compute Phi(x) = log((exp(x) - 1) / x), with Phi(0) = 0.

    Equation (3.8) of the paper.

    Parameters
    ----------
    x : array_like
        Input values (scalar or any shape).

    Returns
    -------
    np.ndarray
        Phi(x), same shape as ``x``.

    Notes
    -----
    Three regimes ensure numerical stability across the full real line:

    * ``|x| < 1e-3`` — Taylor expansion  ``x/2 + x²/24``  avoids
      catastrophic cancellation in ``exp(x) - 1 - x`` near zero.
    * ``x > 500``    — asymptotic form ``x − log(x)`` avoids
      ``expm1`` overflow (overflow threshold ≈ 709 in float64).
    * Otherwise      — ``log(|expm1(x)| / |x|)`` is safe because
      ``expm1(x)`` and ``x`` always share the same sign for ``x ≠ 0``,
      and ``expm1`` handles large-negative ``x`` by returning exactly
      ``−1`` (no underflow of the exponential corrupts the result).

    All branch expressions are guarded by safe substitute values so that
    no branch produces NaN or triggers a RuntimeWarning for inputs
    belonging to a different branch.
    """
    x = np.asarray(x, dtype=float)

    near_zero = np.abs(x) < _NEAR_ZERO_THRESHOLD
    large_pos  = x > _LARGE_POS_THRESHOLD
    masked     = near_zero | large_pos   # elements that bypass the "other" branch

    # ── "other" branch: log(|expm1(x)| / |x|) ──────────────────────────────
    # Prevent expm1 overflow for large-positive x by substituting 0.
    safe_for_expm1 = np.where(large_pos, 0.0, x)
    abs_expm1      = np.abs(np.expm1(safe_for_expm1))

    # For masked elements substitute 1.0 so log(1/1) = 0 (never selected).
    abs_expm1_safe = np.where(masked, 1.0, abs_expm1)
    abs_x_safe     = np.where(masked, 1.0, np.abs(x))

    # ── large-positive branch: x − log(x) ───────────────────────────────────
    # Use x only where large_pos is True (x > 500 > 0); elsewhere use 1.0
    # to keep log argument positive and avoid invalid-value warnings.
    safe_x_log = np.where(large_pos, x, 1.0)

    return np.where(
        near_zero,
        # Taylor:  Phi(x) = x/2 + x²/24 + O(x⁴)
        x * (0.5 + x / 24.0),
        np.where(
            large_pos,
            # Asymptotic: log((e^x − 1)/x) ≈ x − log(x) for x ≫ 1
            x - np.log(safe_x_log),
            # General:  log(|expm1(x)| / |x|)
            np.log(abs_expm1_safe / abs_x_safe),
        ),
    )


def psi(x: np.ndarray) -> np.ndarray:
    """
    Compute Psi(x) = Phi(x) / x, with Psi(0) = 1/2.

    Equation (3.8) of the paper.

    Parameters
    ----------
    x : array_like
        Input values (scalar or any shape).

    Returns
    -------
    np.ndarray
        Psi(x), same shape as ``x``.  Values lie in (0, 1] for all x,
        with Psi(0) = 1/2, Psi(x) → 1 as x → +∞, Psi(x) → 0 as x → −∞.

    Notes
    -----
    Stability regions mirror :func:`phi`:

    * ``|x| < 1e-3`` — Taylor expansion  ``1/2 + x/24``.
    * ``x > 500``    — ``1 − log(x)/x``.
    * Otherwise      — ``log(|expm1(x)| / |x|) / x``.

    Safe substitute values are used in every branch to prevent
    RuntimeWarnings from the unevaluated-but-computed parts of np.where.
    """
    x = np.asarray(x, dtype=float)

    near_zero = np.abs(x) < _NEAR_ZERO_THRESHOLD
    large_pos  = x > _LARGE_POS_THRESHOLD
    masked     = near_zero | large_pos

    # ── "other" branch: log(|expm1(x)| / |x|) / x ──────────────────────────
    safe_for_expm1 = np.where(large_pos, 0.0, x)
    abs_expm1      = np.abs(np.expm1(safe_for_expm1))
    abs_expm1_safe = np.where(masked, 1.0, abs_expm1)  # avoid log(0)
    abs_x_safe     = np.where(masked, 1.0, np.abs(x))  # avoid log(0) denominator
    x_safe         = np.where(masked, 1.0, x)           # avoid division by 0

    phi_over_x = np.log(abs_expm1_safe / abs_x_safe) / x_safe

    # ── large-positive branch: 1 − log(x)/x ─────────────────────────────────
    # Use x only where large_pos is True to keep log argument positive.
    safe_x_log  = np.where(large_pos, x, 1.0)
    safe_x_denom = np.where(large_pos, x, 1.0)   # x > 500, safe divisor

    return np.where(
        near_zero,
        # Taylor: Psi(x) = 1/2 + x/24 + O(x³)
        0.5 + x / 24.0,
        np.where(
            large_pos,
            # Asymptotic: (x − log x)/x = 1 − log(x)/x
            1.0 - np.log(safe_x_log) / safe_x_denom,
            phi_over_x,
        ),
    )


def gibbs_mean(z: np.ndarray) -> np.ndarray:
    """
    Compute the mean of the Gibbs density over u ∈ [0, 1].

    The Gibbs (tilted-uniform) density is

        π*(u | z) = [z / (exp(z) − 1)] · exp(z·u),   u ∈ [0, 1],

    where z = (P − V_plus) / λ.  Its mean is

        μ_π(z) = E_π[U]
                = (z·exp(z) − exp(z) + 1) / (z·(exp(z) − 1))
                = (z − 1 + exp(−z)) / (z·(1 − exp(−z)))   [numerically preferred]

    Parameters
    ----------
    z : array_like
        Argument (scalar or any shape); can be any real number.

    Returns
    -------
    np.ndarray
        μ_π(z) ∈ (0, 1), same shape as ``z``.
        Boundary cases: μ_π(0) = 1/2; μ_π → 1 as z → +∞;
        μ_π → 0 as z → −∞.

    Notes
    -----
    Stability regions:

    * ``|z| < 1e-3``  — Taylor expansion  ``1/2 + z/12``.
    * ``z > 50``      — asymptotic ``1 − 1/z``  (density collapses to u = 1).
    * ``z < −50``     — asymptotic ``−1/z``      (density collapses to u = 0);
                        avoids ``exp(−z) = exp(|z|)`` overflow.
    * Otherwise       — stable form using ``exp(−z)`` only (|z| ≤ 50 keeps
                        ``exp(−z) ≤ exp(50) ≈ 5.2 × 10²¹``, safely in float64).
    """
    z = np.asarray(z, dtype=float)

    near_zero  = np.abs(z) < _NEAR_ZERO_THRESHOLD
    large_pos  = z > 50.0
    large_neg  = z < -50.0
    moderate   = ~near_zero & ~large_pos & ~large_neg

    # ── moderate branch: (z − 1 + exp(−z)) / (z·(1 − exp(−z))) ─────────────
    # Substitute 1.0 in non-moderate slots to prevent exp(-z) overflow.
    safe_z = np.where(moderate, z, 1.0)
    em_z   = np.exp(-safe_z)                   # exp(−z), safe for |z| ≤ 50
    denom  = safe_z * (1.0 - em_z)
    # Guard against tiny denom for the substitute value (safe_z=1 gives
    # denom ≈ 0.632, so this check is mostly defensive).
    denom_safe   = np.where(np.abs(denom) < 1e-300, 1.0, denom)
    moderate_val = (safe_z - 1.0 + em_z) / denom_safe

    # ── large-pos branch: 1 − 1/z ───────────────────────────────────────────
    # Guard z = 0 (near_zero) from causing a divide-by-zero warning.
    z_safe_lp = np.where(large_pos, z, 1.0)

    # ── large-neg branch: −1/z ──────────────────────────────────────────────
    z_safe_ln = np.where(large_neg, z, 1.0)

    return np.where(
        near_zero,
        # Taylor: μ_π(z) = 1/2 + z/12 + O(z³)
        0.5 + z / 12.0,
        np.where(
            large_pos,
            # Density concentrates at u = 1
            1.0 - 1.0 / z_safe_lp,
            np.where(
                large_neg,
                # Density concentrates at u = 0
                -1.0 / z_safe_ln,
                moderate_val,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Regularised value functions
# ---------------------------------------------------------------------------

def regularized_value_at_date(
    V_plus: np.ndarray,
    P: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Compute the entropy-regularised option value at an exercise date.

    Implements

        V_t^λ = V_{t+}^λ + λ · Φ((P_t − V_{t+}^λ) / λ)
              = V_{t+}^λ + (P_t − V_{t+}^λ) · Ψ((P_t − V_{t+}^λ) / λ)

    using the :func:`psi` formulation (more numerically stable than
    computing :func:`phi` and then multiplying by λ).

    Parameters
    ----------
    V_plus : np.ndarray, shape (n_paths,)
        Continuation (post-date) value  V_{t+}^λ  for each path.
    P : np.ndarray, shape (n_paths,)
        Immediate exercise payoff  P_t  for each path.
    lam : float
        Entropy-regularisation parameter λ > 0.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Regularised value  V_t^λ  at the exercise date.

    Notes
    -----
    Special cases:
    * P = V_plus  → z = 0, Ψ(0) = 1/2  → V = V_plus (no adjustment).
    * P ≫ V_plus  → z → +∞, Ψ → 1     → V → P  (exercise is optimal).
    * P ≪ V_plus  → z → −∞, Ψ → 0     → V → V_plus (continuation optimal).
    """
    V_plus = np.asarray(V_plus, dtype=float)
    P      = np.asarray(P,      dtype=float)
    assert V_plus.shape == P.shape, (
        f"V_plus and P must have the same shape; got {V_plus.shape} vs {P.shape}"
    )

    z = (P - V_plus) / lam
    return V_plus + (P - V_plus) * psi(z)


def v_hat(
    cont_value: np.ndarray,
    P: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Compute the full regularised value v^{λ,η}(t, X) at an exercise date.

    Alias for :func:`regularized_value_at_date` used in the TD-learning
    solver.  The formula is identical:

        v̂(cont_value, P, λ) = cont_value + (P − cont_value) · Ψ((P − cont_value) / λ)

    Parameters
    ----------
    cont_value : np.ndarray, shape (n_paths,)
        Parameterised continuation value  V_{t+}^λ  (the "script V" in the
        paper) evaluated on the current batch of paths.
    P : np.ndarray, shape (n_paths,)
        Immediate exercise payoff at the current exercise date.
    lam : float
        Entropy-regularisation parameter λ > 0.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Full regularised value including the exercise-date jump.
    """
    return regularized_value_at_date(cont_value, P, lam)
