"""
Policy iteration algorithm for entropy-regularised Bermudan option pricing.

Implements the Gibbs-policy improvement loop described in the paper.
Each iteration trains new continuation networks via backward regression and
accumulates the time-0 option price.
"""

import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src.special_functions import phi, psi, gibbs_mean
from src.networks import ContinuationNet, train_network
from src.bs_european import (
    bs_put_continuation,
    stulz_maxcall_2d,
    bs_call_price_vec,
)


def _ridge_regression(
    log_X: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    extra_features: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit Ridge regression with polynomial features and return in-sample predictions.

    Parameters
    ----------
    log_X : np.ndarray, shape (n_paths, d)
        Log-moneyness feature matrix log(X / K).
    y : np.ndarray, shape (n_paths,)
        Regression target (continuation value).
    degree : int
        Polynomial feature degree (default 2).
    extra_features : np.ndarray, shape (n_paths, k) or None
        Optional analytic features (Stulz price, BS call prices) appended after
        polynomial expansion, reducing approximation bias significantly.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        In-sample predictions.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(log_X)
    if extra_features is not None:
        X_fit = np.column_stack([X_poly, extra_features])
    else:
        X_fit = X_poly
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_fit, y)
    return ridge.predict(X_fit).astype(np.float64)


class PolicyIterationSolver:
    """
    Policy improvement algorithm for entropy-regularised Bermudan options.

    Algorithm
    ---------
    0. Initialise  V^{λ,0}_{t_i+}  at every exercise date:

       * 1-D Bermudan put  (use_bs=True) :
         V^{λ,0}_{t_i+} = e^{−r·t_i} · BS_put(X_{t_i}, K, r, σ, T−t_i)
       * Multi-asset / general  (use_bs=False) :
         Regress  P_T  on  X_{t_i}  with a small network.

    1. Record  v_0^{λ,0}  computed from the initialisation.

    2. For  n = 0, 1, …, max_iter − 1:

       Backward sweep over j = N, N−1, …, 0:

         a. Train net_j to predict the next-date value from  X_{t_j}:
              V^{n+1}_{t_j+} ≈ E[next_date_value | X_{t_j}]

         b. Compute the policy-corrected value at date j:
              z_old  = (P_{t_j} − V^n_{t_j+}) / λ
              G_j    = λ·Φ(z_old) + (V^n_{t_j+} − V^{n+1}_{t_j+}) · μ_π(z_old)
              V^{n+1}_{t_j} = V^{n+1}_{t_j+} + G_j

         c. Pass  V^{n+1}_{t_j}  as target for date  j−1.

       Record  v_0^{λ,n+1} = mean(V^{n+1}_{t_0}).

    Parameters
    ----------
    lam : float
        Entropy-regularisation parameter λ > 0.
    n_exercise_dates : int
        Number of pre-maturity exercise dates N.  Total grid: N+2 points.
    state_dim : int
        Dimension of the state process.
    hidden_width, depth, n_epochs, lr, device :
        Hyper-parameters forwarded to ContinuationNet / train_network.
    use_ridge : bool
        If True (default), use Ridge regression with polynomial features for
        continuation-value fitting instead of a neural network.  This gives
        exact one-shot convergence and is recommended for d ≤ 10.  For d > 10
        (high-dimensional state), set to False and use the neural network with
        sufficient epochs.
    ridge_degree : int
        Polynomial degree for Ridge regression features (default 2).
    """

    def __init__(
        self,
        lam: float,
        n_exercise_dates: int,
        state_dim: int = 1,
        hidden_width: int = 64,
        depth: int = 3,
        n_epochs: int = 300,
        lr: float = 1e-3,
        device: str = 'cpu',
        use_ridge: bool = True,
        ridge_degree: int = 2,
    ):
        self.lam = lam
        self.n_exercise_dates = n_exercise_dates
        self.state_dim = state_dim
        self.hidden_width = hidden_width
        self.depth = depth
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.use_ridge = use_ridge
        self.ridge_degree = ridge_degree
        self.v0_sequence: list[float] = []

    def _fit_continuation(
        self,
        log_X: np.ndarray,
        y: np.ndarray,
        extra_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit a continuation-value approximation and return in-sample predictions.

        Delegates to Ridge regression for d ≤ 10 or when use_ridge=True;
        falls back to a neural network otherwise.

        Parameters
        ----------
        log_X : np.ndarray, shape (n_paths, d)
            Log-moneyness features  log(X / K).
        y : np.ndarray, shape (n_paths,)
            Regression target (continuation value).
        extra_features : np.ndarray, shape (n_paths, k) or None
            Optional analytic features (e.g. Stulz / BS call prices) that are
            appended to the polynomial-expanded log_X before fitting Ridge.
            For neural-network fallback, these are prepended to the raw input.

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        # Constant-input bypass
        if np.std(log_X) < 1e-6:
            return np.full(len(y), float(np.mean(y)), dtype=np.float64)

        if self.use_ridge:
            return _ridge_regression(
                log_X, y,
                degree=self.ridge_degree,
                extra_features=extra_features,
            )

        # Neural-network fallback (high-dimensional states d > 10)
        if extra_features is not None:
            nn_input = np.column_stack([log_X, extra_features])
        else:
            nn_input = log_X
        input_dim = nn_input.shape[1]
        net = ContinuationNet(input_dim, self.hidden_width, self.depth)
        X_tensor = torch.tensor(nn_input, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        with torch.no_grad():
            net.net[-1].bias.fill_(float(np.mean(y)))
        train_network(
            net, X_tensor, y_tensor,
            n_epochs=self.n_epochs,
            lr=self.lr,
            device=self.device,
            verbose=False,
        )
        with torch.no_grad():
            return net(X_tensor.to(self.device)).cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _analytic_features(
        self,
        X_j: np.ndarray,
        K: float,
        r: float,
        delta: float,
        sigma: float,
        tau_j: float,
    ) -> "np.ndarray | None":
        """
        Build analytic-formula features for Ridge / NN regression at date j.

        d==1 (put):  BS European put price at t_j.
        d==2 (max-call): Stulz exact 2-asset European max-call at t_j.
        d>=10: per-asset BS call prices (d values) + BS call on max asset.

        Returns None at the terminal date (tau_j == 0).
        """
        if tau_j <= 1e-9:
            return None

        if self.state_dim == 1:
            from scipy.stats import norm as _norm
            S_j = X_j[:, 0]
            sq = sigma * np.sqrt(tau_j)
            d1 = (np.log(np.maximum(S_j, 1e-15) / K) + (r + 0.5 * sigma**2) * tau_j) / sq
            d2 = d1 - sq
            feat = K * np.exp(-r * tau_j) * _norm.cdf(-d2) - S_j * _norm.cdf(-d1)
            return np.maximum(feat, 0.0).reshape(-1, 1)

        if self.state_dim == 2:
            S1, S2 = X_j[:, 0], X_j[:, 1]
            return stulz_maxcall_2d(S1, S2, K, r, delta, sigma, tau_j).reshape(-1, 1)

        # d >= 10
        calls = np.column_stack([
            bs_call_price_vec(X_j[:, i], K, r, delta, sigma, tau_j)
            for i in range(self.state_dim)
        ])
        max_S = X_j.max(axis=1)
        max_call = bs_call_price_vec(max_S, K, r, delta, sigma, tau_j)
        return np.column_stack([calls, max_call.reshape(-1, 1)])

    def _initialize_v0(
        self,
        paths_at_exercise_dates: np.ndarray,
        payoffs_at_exercise_dates: np.ndarray,
        exercise_times: list[float],
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.1,
        delta: float = 0.0,
        T: float = 1.0,
        use_bs: bool = True,
    ) -> list[np.ndarray]:
        """
        Build the initial continuation-value estimates V^{lambda,0}_{t_i+}.

        V^{lambda,0}_t = E[P_T | F_t] for all t.

        d==1: Black-Scholes European put formula (analytic, exact).
        d==2: Stulz (1982) 2-asset European max-call (analytic, exact).
        d>=10: MC regression with analytic BS-call features.

        Returns
        -------
        list of np.ndarray, length N+1
            V_plus_old[j] has shape (n_paths,) and represents
            V^{lambda,0}_{t_j+} discounted to time 0.
        """
        from scipy.stats import norm as _norm
        N = self.n_exercise_dates
        P_T = payoffs_at_exercise_dates[:, N + 1]
        V_plus_list: list[np.ndarray] = []

        for j in range(N + 1):
            t_j  = exercise_times[j]
            tau_j = T - t_j
            X_j  = paths_at_exercise_dates[:, j, :]
            df_j = np.exp(-r * t_j)

            if use_bs and self.state_dim == 1:
                S_j = X_j[:, 0]
                if tau_j > 1e-9:
                    sq = sigma * np.sqrt(tau_j)
                    d1 = (np.log(np.maximum(S_j, 1e-15) / K) + (r + 0.5 * sigma**2) * tau_j) / sq
                    d2 = d1 - sq
                    put_j = K * np.exp(-r * tau_j) * _norm.cdf(-d2) - S_j * _norm.cdf(-d1)
                else:
                    put_j = np.maximum(K - S_j, 0.0)
                V_j = df_j * np.maximum(put_j, 0.0)

            elif use_bs and self.state_dim == 2:
                # Stulz exact 2-asset European max-call; stulz output is price at t_j
                S1 = X_j[:, 0]
                S2 = X_j[:, 1]
                V_j = df_j * stulz_maxcall_2d(S1, S2, K, r, delta, sigma, tau_j)

            else:
                log_X_j = np.log(np.maximum(X_j, 1e-8) / K)
                extra   = self._analytic_features(X_j, K, r, delta, sigma, tau_j)
                V_j     = self._fit_continuation(log_X_j, P_T, extra_features=extra)

            V_plus_list.append(V_j)

        return V_plus_list

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(
        self,
        paths_at_exercise_dates: np.ndarray,
        payoffs_at_exercise_dates: np.ndarray,
        exercise_times: list[float],
        max_iter: int = 6,
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.1,
        delta: float = 0.0,
        T: float = 1.0,
        use_bs_init: bool = True,
        verbose: bool = False,
    ) -> list[float]:
        """
        Run the policy-iteration algorithm.

        Parameters
        ----------
        paths_at_exercise_dates : np.ndarray, shape (n_paths, N+2, d)
            State paths.  Column i=0…N are exercise dates; i=N+1 is maturity.
        payoffs_at_exercise_dates : np.ndarray, shape (n_paths, N+2)
            Discounted payoffs at each date.
        exercise_times : list[float]
            [t_0, …, t_N, T].
        max_iter : int
            Number of policy-improvement steps.
        K, r, sigma, T : float
            Option / model parameters (for BS initialisation).
        use_bs_init : bool
            Use BS closed-form initialisation for 1-D put; otherwise MC.
        verbose : bool
            Print iteration summaries.

        Returns
        -------
        list[float]
            Time-0 prices [v_0^{λ,0}, v_0^{λ,1}, …, v_0^{λ,max_iter}].
        """
        N = self.n_exercise_dates
        n_paths = paths_at_exercise_dates.shape[0]
        P_T: np.ndarray = payoffs_at_exercise_dates[:, N + 1].copy()

        # ── Step 0: initialise V_plus_old ──────────────────────────────────
        V_plus_old: list[np.ndarray] = self._initialize_v0(
            paths_at_exercise_dates,
            payoffs_at_exercise_dates,
            exercise_times,
            K=K, r=r, sigma=sigma, delta=delta, T=T,
            use_bs=use_bs_init,
        )

        # ── Record v_0^{λ,0} from initialisation ───────────────────────────
        # V^{λ,0}_t = E[P_T | F_t] is the European continuation process.
        # At t_0 = 0 with X_0 constant this equals the European option price.
        v0_init = float(np.mean(V_plus_old[0]))
        self.v0_sequence = [v0_init]

        if verbose:
            print(f"[PolicyIter] iter 0 (init): v0 = {v0_init:.6f}")

        # ── Policy-iteration steps ─────────────────────────────────────────
        for n in range(max_iter):
            # ── (a) Compute Gibbs stopping probabilities from V_plus_old ──
            # psi_j = Ψ((P_j - V_{t_j+}^{λ,n}) / λ)  at each exercise date j
            psi_j: list[np.ndarray] = []
            for j in range(N + 1):
                P_j = payoffs_at_exercise_dates[:, j]
                z_j = (P_j - V_plus_old[j]) / self.lam
                psi_j.append(np.clip(psi(z_j), 0.0, 1.0))

            # ── (b) Forward sweep: v_0^{λ,n+1} via Gibbs stopping payoff ──
            # E[payoff under Gibbs policy π^{n+1}] where at each date j
            # (given survival), the holder stops with probability psi_j.
            survival: np.ndarray = np.ones(n_paths)
            ep: np.ndarray = np.zeros(n_paths)
            for j in range(N + 1):
                ep += survival * psi_j[j] * payoffs_at_exercise_dates[:, j]
                survival *= (1.0 - psi_j[j])
            ep += survival * P_T
            self.v0_sequence.append(float(np.mean(ep)))

            # ── (c) Backward sweep: update V_plus for next iteration ───────
            # V_full[j] = psi_j * P_j + (1 - psi_j) * V_plus_new[j]
            # V_plus_new[j] = E[V_full[j+1] | X_j]  (regression)
            V_full_next: np.ndarray = P_T.copy()
            V_plus_new: list[np.ndarray | None] = [None] * (N + 1)

            for j in range(N, -1, -1):
                X_j = paths_at_exercise_dates[:, j, :]
                P_j = payoffs_at_exercise_dates[:, j]
                log_X_j = np.log(np.maximum(X_j, 1e-8) / K)
                tau_j = T - exercise_times[j]
                extra_j = self._analytic_features(X_j, K, r, delta, sigma, tau_j)

                # Regression: V_plus_new[j] = E[V_full[j+1] | X_j]
                V_new_plus = self._fit_continuation(
                    log_X_j, V_full_next, extra_features=extra_j
                )
                V_plus_new[j] = V_new_plus

                # V_full[j] becomes the regression target for date j-1
                V_full_next = psi_j[j] * P_j + (1.0 - psi_j[j]) * V_new_plus

            V_plus_old = V_plus_new  # type: ignore[assignment]

            if verbose:
                print(f"[PolicyIter] iter {n + 1}: v0 = {self.v0_sequence[-1]:.6f}")

        return self.v0_sequence

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def time0_price_sequence(self) -> list[float]:
        """Return the recorded sequence of time-0 prices across iterations."""
        return self.v0_sequence
