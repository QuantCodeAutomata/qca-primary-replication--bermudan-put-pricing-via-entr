"""
Black-Scholes analytical prices for European options and a Monte Carlo
approximation for the multi-asset European max-call used as a benchmark
in the Bermudan pricing experiments.

Includes the Stulz (1982) exact analytic formula for the 2-asset European
max-call with dividends, used for conditioning values in the BSDE recursion.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal as _mvn_dist


def bs_put_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """
    Black-Scholes price of a European put option.

    Parameters
    ----------
    S : float
        Current asset price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Asset volatility.
    T : float
        Time to maturity (years).

    Returns
    -------
    float
        European put price  P = K·e^{−rT}·N(−d₂) − S·N(−d₁).

    Notes
    -----
    For T = 0 the formula is handled by returning the intrinsic value
    ``max(K − S, 0)`` directly to avoid division by zero.
    """
    if T <= 0.0:
        return float(max(K - S, 0.0))

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_put_continuation(
    S: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float,
    t: float,
) -> np.ndarray:
    """
    Black-Scholes European put *continuation* value at time t.

    This equals the European put price with remaining time-to-maturity
    τ = T − t, evaluated at each asset price in ``S``.

    Parameters
    ----------
    S : np.ndarray, shape (n_paths,) or broadcastable
        Current asset prices.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Asset volatility.
    T : float
        Option expiry (years from time 0).
    t : float
        Current time (years from time 0); must satisfy 0 ≤ t ≤ T.

    Returns
    -------
    np.ndarray
        European put prices at each asset price in ``S``, same shape as
        input.  At expiry (t = T) the intrinsic value is returned.
    """
    S = np.asarray(S, dtype=float)
    tau = T - t

    if tau <= 0.0:
        return np.maximum(K - S, 0.0)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_european_maxcall_price_approx(
    x0: float,
    K: float,
    r: float,
    sigma: float,
    delta: float,
    T: float,
    d: int,
    n_paths: int = 200_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo price of a European max-call on ``d`` independent assets.

    The payoff at maturity is

        Payoff = exp(−r·T) · max(max_i X_T^i − K, 0)

    where each X_T^i follows a log-normal distribution with parameters
    (x0, r, sigma, delta).

    Parameters
    ----------
    x0 : float
        Initial price for every asset.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility (same for all assets).
    delta : float
        Continuous dividend yield (same for all assets).
    T : float
        Time to maturity (years).
    d : int
        Number of independent assets.
    n_paths : int
        Number of Monte Carlo paths.  Default 200 000.
    seed : int
        RNG seed.  Default 42.

    Returns
    -------
    float
        Monte Carlo estimate of the European max-call price.

    Notes
    -----
    A single-step simulation is used (no time discretisation error for
    European options):  X_T^i = x0 · exp((r − δ − σ²/2)·T + σ·√T·Z_i)
    where Z_i ~ N(0, 1) are independent across assets and paths.
    """
    rng = np.random.default_rng(seed)

    # Draw terminal log-normal values directly (one Euler step = exact for GBM)
    Z = rng.standard_normal((n_paths, d))           # shape (n_paths, d)
    log_drift = (r - delta - 0.5 * sigma ** 2) * T
    log_diff  = sigma * np.sqrt(T) * Z
    S_T = x0 * np.exp(log_drift + log_diff)         # shape (n_paths, d)

    max_asset = S_T.max(axis=1)                     # shape (n_paths,)
    payoffs   = np.exp(-r * T) * np.maximum(max_asset - K, 0.0)

    return float(payoffs.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Stulz (1982) exact analytic formula for the 2-asset European max-call
# ──────────────────────────────────────────────────────────────────────────────

def _bvn_cdf(a: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    """
    Vectorized bivariate standard normal CDF P(Z1 ≤ a, Z2 ≤ b; rho).

    Uses scipy.stats.multivariate_normal; accurate and fast enough for
    path counts up to ~50 000 (≈ 0.1–0.2 s for 20 000 points).

    Parameters
    ----------
    a, b : np.ndarray, shape (n,)
        Upper integration limits.
    rho : float
        Pearson correlation between Z1 and Z2.

    Returns
    -------
    np.ndarray, shape (n,)
        P(Z1 ≤ a, Z2 ≤ b).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    pts = np.column_stack([a, b])
    cov = [[1.0, rho], [rho, 1.0]]
    return _mvn_dist(mean=[0.0, 0.0], cov=cov).cdf(pts)


def stulz_maxcall_2d(
    S1: np.ndarray,
    S2: np.ndarray,
    K: float,
    r: float,
    delta: float,
    sigma: float,
    tau: float,
) -> np.ndarray:
    """
    Stulz (1982) exact price of a European max-call on two independent assets.

    Computes  C_max = E[e^{-rτ} · max(max(S1_T, S2_T) − K, 0)]  where the two
    assets follow independent GBM with the same dividend yield δ and volatility σ:
        S_T^i = S_i · exp((r − δ − σ²/2)τ + σ√τ · Zi),  Z1, Z2 iid N(0,1).

    The closed-form follows from Stulz (1982, eq. 3.1) specialized to ρ = 0:

        C_max = C_BS(S1,K) + C_BS(S2,K) − C_exchange(S1,S2)

    where C_BS(Si,K) is the individual Black-Scholes call and
    C_exchange is the Margrabe exchange option price.

    The equivalent bivariate-normal form used here (rho = +1/√2 in the
    adjusted moneyness variables) follows directly from Stulz (1982) eq. (14):

        C_max = S1·e^{-δτ}·N2(d1_1, h12; ρ) + S2·e^{-δτ}·N2(d1_2, h21; ρ)
                − K·e^{-rτ}·[N2(d2_1, g12; ρ) + N2(d2_2, g21; ρ)]

    where ρ = 1/√2  (the correlation of d1_1 and d1_2 that arises from the
    symmetry of the bivariate log-normal with independent Brownian drivers),
    and the arguments are:

        d1_i = [log(Si/K) + (r − δ + σ²/2)τ] / (σ√τ)
        d2_i = d1_i − σ√τ
        h12  = [log(S1/S2) + σ²τ] / (σ√(2τ))
        h21  = [log(S2/S1) + σ²τ] / (σ√(2τ))
        g12  =  log(S1/S2) / (σ√(2τ))
        g21  =  log(S2/S1) / (σ√(2τ))

    The price returned is undiscounted (expressed as a present value using the
    discounting factor e^{-rτ} already embedded in the formula).

    Parameters
    ----------
    S1, S2 : np.ndarray, shape (n,)
        Current prices of asset 1 and asset 2.
    K : float
        Common strike price.
    r : float
        Continuously compounded risk-free rate.
    delta : float
        Common continuous dividend yield.
    sigma : float
        Common asset volatility.
    tau : float
        Time to maturity.  If ≤ 0 the intrinsic value is returned.

    Returns
    -------
    np.ndarray, shape (n,)
        European max-call price for each path, expressed as a time-0 value
        (NOT multiplied by any additional discount factor).

    Notes
    -----
    The correlation ρ = +1/√2 (NOT −1/√2) arises from the construction of
    the bivariate normal for two independent log-normal assets; it equals
    the correlation of their respective d1 statistics relative to the
    combined volatility σ√(2τ).  See Stulz (1982) appendix for derivation.
    """
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)

    if tau <= 1e-9:
        return np.maximum(np.maximum(S1, S2) - K, 0.0)

    rho = 1.0 / np.sqrt(2.0)   # correlation parameter for ρ=0 independent assets
    sq_tau  = sigma * np.sqrt(tau)
    sq_2tau = sigma * np.sqrt(2.0 * tau)

    log_S1_K = np.log(np.maximum(S1, 1e-15) / K)
    log_S2_K = np.log(np.maximum(S2, 1e-15) / K)
    log_S1_S2 = np.log(np.maximum(S1, 1e-15) / np.maximum(S2, 1e-15))

    d1_1 = (log_S1_K + (r - delta + 0.5 * sigma ** 2) * tau) / sq_tau
    d1_2 = (log_S2_K + (r - delta + 0.5 * sigma ** 2) * tau) / sq_tau
    d2_1 = d1_1 - sq_tau
    d2_2 = d1_2 - sq_tau

    h12 = ( log_S1_S2 + sigma ** 2 * tau) / sq_2tau
    h21 = (-log_S1_S2 + sigma ** 2 * tau) / sq_2tau
    g12 =  log_S1_S2 / sq_2tau
    g21 = -log_S1_S2 / sq_2tau

    dS = np.exp(-delta * tau)
    dK = np.exp(-r    * tau)

    n2a = _bvn_cdf(d1_1, h12, rho)
    n2b = _bvn_cdf(d1_2, h21, rho)
    n2c = _bvn_cdf(d2_1, g12, rho)
    n2d = _bvn_cdf(d2_2, g21, rho)

    price = S1 * dS * n2a + S2 * dS * n2b - K * dK * (n2c + n2d)
    return np.maximum(price, 0.0)


def bs_call_price_vec(
    S: np.ndarray,
    K: float,
    r: float,
    delta: float,
    sigma: float,
    tau: float,
) -> np.ndarray:
    """
    Vectorized Black-Scholes call price for a single asset with dividends.

    Parameters
    ----------
    S : np.ndarray, shape (n,)
        Current asset prices.
    K, r, delta, sigma, tau : float
        Strike, risk-free rate, dividend yield, volatility, time to maturity.

    Returns
    -------
    np.ndarray, shape (n,)
        European call prices.
    """
    S = np.asarray(S, dtype=float)
    if tau <= 1e-9:
        return np.maximum(S - K, 0.0)
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(np.maximum(S, 1e-15) / K) + (r - delta + 0.5 * sigma ** 2) * tau) / sq
    d2 = d1 - sq
    return np.maximum(
        S * np.exp(-delta * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2),
        0.0,
    )
