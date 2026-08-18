# Entropy-Regularized BSDE Approach to Bermudan Options

A Python implementation of the numerical experiments accompanying the paper:

> **An Entropy Regularized BSDE Approach to Bermudan Options and Games**

The method replaces the discrete optimal-stopping problem underlying Bermudan
option pricing with a family of smoothed problems parameterised by an entropy
regularisation coefficient λ.  As λ → 0 the regularised price converges to
the true Bermudan price.  Two complementary solvers are implemented:

* **Policy iteration** – a backward-induction scheme that converges
  super-linearly in the number of exercise dates.
* **TD (temporal-difference) learning** – a forward-pass solver that is
  amenable to neural-network function approximation and scales to high
  dimensions.

---

## Repository Layout

```
project/
├── data/               # (empty placeholder for any external data)
├── exp/                # Runnable experiment scripts
│   ├── exp1_bermudan_put.py
│   ├── exp2_bermudan_put_convergence.py
│   ├── exp3_bermudan_maxcall.py
│   ├── exp4_bermudan_maxcall_highdim.py
│   ├── exp5_game_option.py
│   └── exp6_sensitivity.py
├── results/            # Output tables, figures, and RESULTS.md
├── src/                # Core library modules
│   ├── __init__.py
│   ├── bs_european.py      # Black-Scholes European formulas
│   ├── reporting.py        # Table formatting and CSV export
│   ├── simulator.py        # Black-Scholes Monte Carlo path simulation
│   └── special_functions.py# Φ, Ψ, Gibbs-mean, regularised value
├── tests/              # Unit and integration tests
├── requirements.txt
└── README.md
```

---

## Experiments

| Script | Description |
|--------|-------------|
| `exp/exp1_bermudan_put.py` | 1-D Bermudan put — policy iteration vs. TD price table (reproduces Table 1) |
| `exp/exp2_bermudan_put_convergence.py` | Convergence study: regularised price vs. λ for the 1-D put |
| `exp/exp3_bermudan_maxcall.py` | Multi-asset Bermudan max-call (d = 2, 3, 5) — policy iteration |
| `exp/exp4_bermudan_maxcall_highdim.py` | High-dimensional max-call (d = 10, 20) — TD solver with neural network |
| `exp/exp5_game_option.py` | Bermudan game option (stopper vs. controller) |
| `exp/exp6_sensitivity.py` | Sensitivity analysis: effect of λ, batch size, and network width |

---

## How to Run

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Run any experiment script directly
python exp/exp1_bermudan_put.py
python exp/exp2_bermudan_put_convergence.py
python exp/exp3_bermudan_maxcall.py
python exp/exp4_bermudan_maxcall_highdim.py
python exp/exp5_game_option.py
python exp/exp6_sensitivity.py
```

Results (tables, figures, summary text) are written to the `results/`
directory.  A running summary is appended to `results/RESULTS.md` each time
an experiment completes.

---

## Requirements

| Package | Version |
|---------|---------|
| torch | ≥ 2.0 |
| numpy | ≥ 1.26 |
| scipy | ≥ 1.14 |
| pandas | ≥ 2.0 |
| matplotlib | latest |
| seaborn | latest |
| pytest | latest |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Key Mathematical Objects

### Special functions

* **Φ(x)** = log((eˣ − 1) / x), with Φ(0) = 0 — measures the log-partition
  function deviation; see `src/special_functions.py:phi`.
* **Ψ(x)** = Φ(x) / x, with Ψ(0) = 1/2 — soft exercise weight; see
  `src/special_functions.py:psi`.
* **μ_π(z)** = E[U | U ∼ Gibbs(z)] — mean of the tilted-uniform density;
  see `src/special_functions.py:gibbs_mean`.

### Regularised value update

At each exercise date tᵢ the regularised value is updated via

    V_tᵢ^λ = V_{tᵢ₊}^λ + (P_tᵢ − V_{tᵢ₊}^λ) · Ψ((P_tᵢ − V_{tᵢ₊}^λ) / λ)

implemented in `src/special_functions.py:regularized_value_at_date`.
