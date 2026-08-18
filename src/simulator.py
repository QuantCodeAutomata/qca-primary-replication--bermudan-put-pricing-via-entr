"""
Black-Scholes Monte Carlo path simulator for Bermudan option pricing experiments.
"""

import numpy as np


def simulate_bs_paths(
    x0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate 1-D Black-Scholes paths using exact GBM increments.

    Parameters
    ----------
    x0 : float
        Initial asset price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time horizon (years).
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps (intervals).
    seed : int or None
        RNG seed for reproducibility.
    delta : float
        Continuous dividend yield. Default 0.

    Returns
    -------
    times : np.ndarray, shape (n_steps+1,)
        Uniform time grid from 0 to T.
    paths : np.ndarray, shape (n_paths, n_steps+1)
        Simulated asset prices at each time step.

    Notes
    -----
    Dynamics: X_t = x0 * exp((r - delta - sigma^2/2)*t + sigma*W_t)
    Simulation uses exact log-normal increments to avoid discretisation error.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    # Generate standard normal increments: shape (n_paths, n_steps)
    Z = rng.standard_normal((n_paths, n_steps))

    drift_per_step = (r - delta - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z          # shape (n_paths, n_steps)

    # Cumulative log-returns; prepend 0 for t=0
    log_increments = drift_per_step + diffusion   # shape (n_paths, n_steps)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )                                              # shape (n_paths, n_steps+1)

    paths = x0 * np.exp(log_paths)

    assert times.shape == (n_steps + 1,), "times shape mismatch"
    assert paths.shape == (n_paths, n_steps + 1), "paths shape mismatch"
    return times, paths


def simulate_bs_paths_nd(
    x0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    d: int,
    seed: int | None = None,
    delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate d-dimensional independent Black-Scholes paths.

    All d assets share the same parameters (x0, r, sigma, delta) but are
    driven by independent Brownian motions.

    Parameters
    ----------
    x0 : float
        Initial price for every asset.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility (same for all assets).
    T : float
        Time horizon (years).
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.
    d : int
        Number of assets.
    seed : int or None
        RNG seed.
    delta : float
        Continuous dividend yield (same for all assets). Default 0.

    Returns
    -------
    times : np.ndarray, shape (n_steps+1,)
        Uniform time grid from 0 to T.
    paths : np.ndarray, shape (n_paths, n_steps+1, d)
        Simulated asset prices for each path, time step, and asset.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    # Independent Brownian increments for each asset
    Z = rng.standard_normal((n_paths, n_steps, d))  # shape (n_paths, n_steps, d)

    drift_per_step = (r - delta - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z              # shape (n_paths, n_steps, d)

    log_increments = drift_per_step + diffusion
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1, d)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )                                                # shape (n_paths, n_steps+1, d)

    paths = x0 * np.exp(log_paths)

    assert times.shape == (n_steps + 1,), "times shape mismatch"
    assert paths.shape == (n_paths, n_steps + 1, d), "paths shape mismatch"
    return times, paths


def get_exercise_date_indices(
    times: np.ndarray,
    exercise_dates: list[float],
) -> list[int]:
    """
    Map exercise dates to the nearest indices in a time grid.

    Parameters
    ----------
    times : np.ndarray, shape (n_steps+1,)
        Uniform time grid returned by a simulate_* function.
    exercise_dates : list of float
        Desired exercise dates (must lie within [times[0], times[-1]]).

    Returns
    -------
    list of int
        Index in ``times`` closest to each entry in ``exercise_dates``,
        preserving the input order.

    Examples
    --------
    >>> times = np.linspace(0, 1, 11)          # 0.0, 0.1, ..., 1.0
    >>> get_exercise_date_indices(times, [0.5, 1.0])
    [5, 10]
    """
    return [int(np.argmin(np.abs(times - t))) for t in exercise_dates]


def compute_put_payoffs(
    paths: np.ndarray,
    times: np.ndarray,
    K: float,
    r: float,
    exercise_indices: list[int],
) -> np.ndarray:
    """
    Compute discounted European put payoffs at specified exercise dates.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, n_steps+1)
        1-D asset price paths from :func:`simulate_bs_paths`.
    times : np.ndarray, shape (n_steps+1,)
        Corresponding time grid.
    K : float
        Strike price.
    r : float
        Risk-free rate (used for discounting).
    exercise_indices : list of int
        Indices into ``times``/``paths`` at which to evaluate payoffs.

    Returns
    -------
    np.ndarray, shape (n_paths, len(exercise_indices))
        Discounted put payoffs exp(-r*t_i) * max(K - X_{t_i}, 0) for each
        path and exercise date.
    """
    n_paths = paths.shape[0]
    n_ex = len(exercise_indices)
    payoffs = np.empty((n_paths, n_ex), dtype=float)

    for j, idx in enumerate(exercise_indices):
        t = times[idx]
        discount = np.exp(-r * t)
        payoffs[:, j] = discount * np.maximum(K - paths[:, idx], 0.0)

    assert payoffs.shape == (n_paths, n_ex), "payoffs shape mismatch"
    return payoffs


def compute_maxcall_payoffs(
    paths_nd: np.ndarray,
    times: np.ndarray,
    K: float,
    r: float,
    exercise_indices: list[int],
) -> np.ndarray:
    """
    Compute discounted max-call payoffs for d-dimensional asset paths.

    The max-call payoff at time t is exp(-r*t) * max(max_i X_t^i - K, 0),
    where the inner max is taken over assets.

    Parameters
    ----------
    paths_nd : np.ndarray, shape (n_paths, n_steps+1, d)
        Multi-asset price paths from :func:`simulate_bs_paths_nd`.
    times : np.ndarray, shape (n_steps+1,)
        Corresponding time grid.
    K : float
        Strike price.
    r : float
        Risk-free rate (used for discounting).
    exercise_indices : list of int
        Indices into the time grid at which to evaluate payoffs.

    Returns
    -------
    np.ndarray, shape (n_paths, len(exercise_indices))
        Discounted max-call payoffs at each exercise date.
    """
    n_paths = paths_nd.shape[0]
    n_ex = len(exercise_indices)
    payoffs = np.empty((n_paths, n_ex), dtype=float)

    for j, idx in enumerate(exercise_indices):
        t = times[idx]
        discount = np.exp(-r * t)
        max_asset = paths_nd[:, idx, :].max(axis=1)   # shape (n_paths,)
        payoffs[:, j] = discount * np.maximum(max_asset - K, 0.0)

    assert payoffs.shape == (n_paths, n_ex), "payoffs shape mismatch"
    return payoffs
