"""
TD-based BSDE solver for entropy-regularised Bermudan option pricing.

Trains one continuation-value network per exercise date in a backward pass,
using the temporal-difference (TD) target derived from the regularised value
formula at the next date.
"""

import numpy as np
import torch
from scipy.stats import norm as _sp_norm

from src.special_functions import v_hat
from src.networks import ContinuationNet, train_network
from src.bs_european import stulz_maxcall_2d, bs_call_price_vec


class TDSolver:
    """
    TD-based BSDE solver for entropy-regularised Bermudan options.

    Trains date-specific continuation networks backward from maturity.
    At each exercise date t_i (backward, i = N, …, 0):

      - Target : v^{λ, η_{i+1}}(t_{i+1}, X_{t_{i+1}})
      - Trains : script_V^{η_i}(X_{t_i}) to minimise
                 E[(target − script_V^{η_i}(X_{t_i}))²]

    Terminal target: v(T, X_T) = P_T  (discounted payoff at maturity).

    Parameters
    ----------
    lam : float
        Entropy-regularisation parameter λ > 0.
    n_exercise_dates : int
        Number of pre-maturity exercise dates N.
        Total time grid has N + 2 points (indices 0 … N for exercise dates,
        N + 1 for maturity).
    state_dim : int
        Dimension d of the state process X.
    hidden_width : int
        Width of every hidden layer in each ContinuationNet.
    depth : int
        Total number of linear layers in each ContinuationNet.
    n_epochs : int
        Training epochs per network.
    lr : float
        Adam learning rate.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        lam: float,
        n_exercise_dates: int,
        state_dim: int,
        hidden_width: int = 64,
        depth: int = 3,
        n_epochs: int = 500,
        lr: float = 1e-3,
        device: str = 'cpu',
    ):
        self.lam = lam
        self.n_exercise_dates = n_exercise_dates
        self.state_dim = state_dim
        self.hidden_width = hidden_width
        self.depth = depth
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.nets: list[ContinuationNet] = []
        self.K: float = 100.0    # set by fit(); used for log-normalisation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_features(
        self,
        X_i: np.ndarray,
        K: float,
        r: float,
        delta: float,
        sigma: float,
        tau_i: float,
    ) -> np.ndarray:
        """
        Build input feature tensor for the continuation network at date i.

        Augments log-moneyness with analytic European option prices so the
        network starts from a near-optimal basis:
          d==1 : [log(S/K), BS_put(S,tau)]
          d==2 : [log(S1/K), log(S2/K), Stulz_maxcall(S1,S2,tau)]
          d>=10: [log(S/K) per asset, per-asset BS calls, BS call on max]
        """
        log_X_i = np.log(np.maximum(X_i, 1e-8) / K)

        if tau_i <= 1e-9:
            return log_X_i  # terminal date: no option-price feature

        if self.state_dim == 1:
            S = X_i[:, 0]
            sq = sigma * np.sqrt(tau_i)
            d1 = (np.log(np.maximum(S, 1e-15) / K) + (r + 0.5 * sigma ** 2) * tau_i) / sq
            d2 = d1 - sq
            put = np.maximum(
                K * np.exp(-r * tau_i) * _sp_norm.cdf(-d2)
                - S * np.exp(-delta * tau_i) * _sp_norm.cdf(-d1),
                0.0,
            )
            return np.column_stack([log_X_i, put.reshape(-1, 1)])

        if self.state_dim == 2:
            stulz = stulz_maxcall_2d(
                X_i[:, 0], X_i[:, 1], K, r, delta, sigma, tau_i
            )
            return np.column_stack([log_X_i, stulz.reshape(-1, 1)])

        # d >= 10: per-asset BS calls + call on max as summary feature
        calls = np.column_stack([
            bs_call_price_vec(X_i[:, j], K, r, delta, sigma, tau_i)
            for j in range(self.state_dim)
        ])
        max_call = bs_call_price_vec(
            X_i.max(axis=1), K, r, delta, sigma, tau_i
        )
        return np.column_stack([log_X_i, calls, max_call.reshape(-1, 1)])

    def fit(
        self,
        paths_at_exercise_dates: np.ndarray,
        payoffs_at_exercise_dates: np.ndarray,
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.2,
        delta: float = 0.0,
        T: float = 1.0,
        exercise_times: list[float] | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Train all date networks in a single backward sweep.

        Parameters
        ----------
        paths_at_exercise_dates : np.ndarray, shape (n_paths, N+2, d)
            State at each date.  Index i = 0...N are exercise dates;
            index N+1 is maturity.
        payoffs_at_exercise_dates : np.ndarray, shape (n_paths, N+2)
            Discounted payoff P_{t_i} at each date.
        K : float  Strike price.
        r : float  Risk-free rate.
        sigma : float  Asset volatility.
        delta : float  Dividend yield.
        T : float  Option maturity.
        exercise_times : list[float] or None
            [t_0, ..., t_N, T].  Uniform spacing assumed when None.
        verbose : bool
        """
        self.K = K
        N = self.n_exercise_dates

        if exercise_times is None:
            exercise_times = [T * i / (N + 1) for i in range(N + 1)] + [T]

        # Store for inference
        self._r = r
        self._sigma = sigma
        self._delta = delta
        self._T = T
        self._exercise_times = exercise_times

        nets_rev: list[ContinuationNet] = []

        # Terminal TD target: v(T, X_T) = P_T
        current_target: np.ndarray = payoffs_at_exercise_dates[:, N + 1].copy()

        for i in range(N, -1, -1):
            X_i = paths_at_exercise_dates[:, i, :]          # (n_paths, d)
            P_i = payoffs_at_exercise_dates[:, i]            # (n_paths,)
            tau_i = T - exercise_times[i]                    # remaining life

            feats = self._build_features(X_i, K, r, delta, sigma, tau_i)
            X_t = torch.tensor(feats, dtype=torch.float32)
            y_t = torch.tensor(current_target, dtype=torch.float32)

            net_i = ContinuationNet(feats.shape[1], self.hidden_width, self.depth)
            train_network(
                net_i, X_t, y_t,
                n_epochs=self.n_epochs,
                lr=self.lr,
                device=self.device,
                verbose=verbose,
            )
            nets_rev.append(net_i)

            # Build next TD target: v^{lam,eta_i}(t_i, X_{t_i})
            with torch.no_grad():
                script_V = net_i(X_t.to(self.device)).cpu().numpy()   # (n_paths,)

            current_target = v_hat(script_V, P_i, self.lam)

        # Chronological order: nets[0] -> date 0, nets[N] -> date N
        self.nets = list(reversed(nets_rev))

    def predict_continuation(
        self,
        X: np.ndarray,
        date_idx: int,
    ) -> np.ndarray:
        """
        Evaluate script_V^{eta_{date_idx}}(X) for all paths.

        Parameters
        ----------
        X : np.ndarray, shape (n_paths, d)
            Raw asset prices (log-normalised internally via stored params).
        date_idx : int
            Exercise date index in 0...N.

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        net = self.nets[date_idx]
        tau_i = self._T - self._exercise_times[date_idx]
        feats = self._build_features(
            X, self.K, self._r, self._delta, self._sigma, tau_i
        )
        X_t = torch.tensor(feats, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return net(X_t).cpu().numpy()

    def predict_value(
        self,
        X: np.ndarray,
        P: np.ndarray,
        date_idx: int,
    ) -> np.ndarray:
        """
        Evaluate the full regularised value v^{λ,η}(t_i, X).

        v = script_V + (P − script_V) · Ψ((P − script_V) / λ)

        Parameters
        ----------
        X : np.ndarray, shape (n_paths, d)
            Raw asset prices (log-normalised internally).
        P : np.ndarray, shape (n_paths,)
            Discounted payoffs at this date.
        date_idx : int

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        cont = self.predict_continuation(X, date_idx)
        return v_hat(cont, P, self.lam)

    def time0_price(
        self,
        paths_at_exercise_dates: np.ndarray,
        payoffs_at_exercise_dates: np.ndarray,
    ) -> float:
        """
        Estimate the time-0 option price.

        Computes the mean over paths of v^{λ,η_0}(t_0, X_{t_0}).

        Parameters
        ----------
        paths_at_exercise_dates : np.ndarray, shape (n_paths, N+2, d)
        payoffs_at_exercise_dates : np.ndarray, shape (n_paths, N+2)

        Returns
        -------
        float
        """
        X0 = paths_at_exercise_dates[:, 0, :]    # (n_paths, d)
        P0 = payoffs_at_exercise_dates[:, 0]      # (n_paths,)
        values = self.predict_value(X0, P0, date_idx=0)
        return float(np.mean(values))
