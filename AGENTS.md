# Repository Knowledge — Entropy-Regularised Bermudan Options

## Overview
Implementation of "An Entropy Regularized BSDE Approach to Bermudan Options and Games".
Two solvers: policy iteration (backward) and TD learning (forward/neural-network-friendly).

## Project Layout
```
src/                     Core library (all production code)
  __init__.py            Empty package marker
  simulator.py           Black-Scholes GBM simulation (1-D and n-D)
  special_functions.py   Phi, Psi, Gibbs-mean, regularized_value_at_date, v_hat
  bs_european.py         Closed-form BS put + MC max-call benchmark
  reporting.py           Table construction, CSV/TXT export, RESULTS.md append
exp/                     Experiment runner scripts (exp1_…py through exp6_…py)
results/                 Output tables, figures, RESULTS.md (gitignored temp files)
tests/                   Unit & integration tests
```

## Key Mathematical Notes

### phi(x) = log((exp(x)-1)/x)  [Eq 3.8]
- phi(0) = 0  (limit)
- Taylor near 0: x/2 + x²/24  (valid to O(x⁴), used for |x| < 1e-3)
- Large positive x (>500): x − log(x)  (avoids expm1 overflow above ~709)
- All other x: log(|expm1(x)| / |x|)  — expm1 handles large negative x (returns −1)

### psi(x) = phi(x)/x  [Eq 3.8]
- psi(0) = 1/2  (limit)
- Taylor near 0: 1/2 + x/24
- Large positive x: 1 − log(x)/x
- Ranges in (0,1): psi→1 as x→+∞, psi→0 as x→−∞

### gibbs_mean(z) = (z−1+exp(−z)) / (z·(1−exp(−z)))
- Mean of Gibbs/tilted-uniform density π*(u|z) ∝ exp(z·u) on [0,1]
- gibbs_mean(0) = 1/2; →1 as z→+∞; →0 as z→−∞
- Taylor near 0: 1/2 + z/12
- Large positive z (>50): 1 − 1/z  (avoids exp overflow)
- Large negative z (<−50): −1/z = 1/|z|  (avoids exp(|z|) overflow)
- Moderate z: formula using exp(−z) only; exp(50)≈5e21 < float64 max

### regularized_value_at_date(V_plus, P, lam)
  V_t^λ = V_plus + (P − V_plus) · psi((P − V_plus) / lam)
  Uses psi() formulation (numerically preferable to lambda*phi(z/lambda))

### v_hat = alias for regularized_value_at_date (used in TD solver)

## Safety Pattern for np.where
All np.where branches are computed before selection.  To avoid RuntimeWarnings
from branches applied to out-of-range inputs, every function substitutes safe
dummy values (e.g. 1.0 instead of 0) in masked slots before expensive operations
(log, exp, division).  The pattern is:
  safe_x = np.where(active_mask, x, safe_dummy)
  branch_val = expensive_fn(safe_x)
  result = np.where(mask_A, val_A, np.where(mask_B, branch_val, val_C))

## Simulator Conventions
- simulate_bs_paths → (times shape (n_steps+1,), paths shape (n_paths, n_steps+1))
- simulate_bs_paths_nd → (times shape (n_steps+1,), paths shape (n_paths, n_steps+1, d))
- get_exercise_date_indices uses argmin(|times − t|); ties go to the earlier index
- compute_put_payoffs / compute_maxcall_payoffs return shape (n_paths, n_exercise_dates)
- All payoffs are discounted: exp(−r·t) · intrinsic

## New Modules (implemented)

### src/networks.py
- `ContinuationNet(input_dim, hidden_width, depth)` — fully connected net.
  Architecture: `Linear(d,w) → [Tanh → Linear(w,w)] × (depth−2) → Tanh → Linear(w,1)`.
  `forward` returns shape `(batch,)` (last dim squeezed).
- `train_network(net, X, y, n_epochs, lr, batch_size, device, verbose)` — Adam / MSE mini-batch
  trainer; returns `list[float]` loss history (one entry per epoch).

### src/td_solver.py  — TDSolver
Backward TD sweep; trains `N+1` networks (one per exercise date i=0…N).
- Inputs: `paths_at_exercise_dates (n_paths, N+2, d)`, `payoffs_at_exercise_dates (n_paths, N+2)`.
  Index N+1 is maturity; all payoffs discounted to time 0.
- Backward loop starts with terminal target `P_T = payoffs[:,N+1]`.
  At each date i: train `nets[i]` on target, then update target →
  `v_hat(nets[i](X_i), P_i, lam)` for the next (earlier) date.
- `predict_continuation(X, date_idx)` → `(n_paths,)` numpy.
- `predict_value(X, P, date_idx)` → `v_hat(continuation, P, lam)`.
- `time0_price(paths, payoffs)` → `mean(predict_value(..., date_idx=0))`.

### src/policy_iteration.py  — PolicyIterationSolver
Policy improvement; returns `v0_sequence` of length `max_iter+1`.
- Initialisation (`_initialize_v0`):
  * 1-D (use_bs=True): `V_plus_old[j] = exp(−r·t_j) · bs_put_continuation(S_j, K, r, σ, T, t_j)`.
    This equals `E^Q[P_T | X_{t_j}]`, consistent with discounted payoffs.
  * Multi-D: train a small network to regress `P_T` on `X_{t_j}`.
- `v0_sequence[0]`: computed from init as `mean(V_plus_old[0] + λ·Φ(z_0))` where
  `z_0 = (P_0 − V_plus_old[0]) / λ`.
- Each iteration n→n+1: backward sweep j=N…0:
  `G = λ·Φ(z_old) + (V_old − V_new) · gibbs_mean(z_old)`
  `V^{n+1}_{t_j} = V_new_j + G`; pass as target to date j−1.
  After loop, `v0 = mean(V^{n+1}_{t_0})`.
- Old `V_plus` arrays updated to new network predictions after each iteration.

## Running Experiments
  python exp/exp1_bermudan_put.py   # through exp6_sensitivity.py
Results written to results/ (CSV, TXT, RESULTS.md).
