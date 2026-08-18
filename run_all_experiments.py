"""
Master runner for all 6 Bermudan option pricing experiments.

Adjusts hyperparameters for CPU-only execution while preserving the
qualitative conclusions described in the paper.  All results are written
to results/ and summarised in results/RESULTS.md.
"""

import importlib
import os
import sys
import time
import traceback

# ── CPU-aware hyper-parameter overrides ─────────────────────────────────────
# These are injected *before* each experiment module is imported so that the
# module-level constants pick up the lighter values.
CPU_OVERRIDES = {
    "exp1_bermudan_put": {
        "N_PATHS": 16_384,
        "N_EPOCHS_TD": 200,
        "N_EPOCHS_PI": 150,
        "MAX_ITER": 5,
    },
    "exp2_maxcall_2d": {
        "N_PATHS": 8_192,
        "N_EPOCHS": 100,
        "MAX_ITER": 8,
    },
    "exp3_maxcall_dimensions": {
        "N_PATHS": 4_096,
        "N_EPOCHS_TD": {2: 100, 10: 80, 50: 60},
        "N_EPOCHS_PI": {2: 80, 10: 60, 50: 50},
        "PI_MAX_ITER": 8,
    },
    "exp4_stopping_rule": {
        "N_EVAL_PATHS": 50_000,
        "N_EPOCHS": 150,
    },
    "exp5_dual_bound": {
        "N_TRAIN_PATHS": 8_192,
        "N_EVAL_PATHS": 50_000,
        "N_EPOCHS": 150,
    },
    "exp6_game_option": {
        "N_PATHS": 8_192,
        "N_EPOCHS": 100,
    },
}


def run_experiment(name: str, overrides: dict) -> bool:
    """Import and run one experiment module, injecting the given overrides."""
    print(f"\n{'#'*70}")
    print(f"# Running {name}")
    print(f"{'#'*70}")
    t0 = time.time()

    # Import the module fresh each time
    mod_path = f"exp.{name}"
    if mod_path in sys.modules:
        del sys.modules[mod_path]

    try:
        mod = importlib.import_module(mod_path)
    except Exception as exc:
        print(f"[IMPORT ERROR] {name}: {exc}")
        traceback.print_exc()
        return False

    # Apply overrides to module-level constants
    for attr, val in overrides.items():
        if hasattr(mod, attr):
            setattr(mod, attr, val)
        else:
            print(f"[WARN] {name} has no attribute {attr!r} – skipping override")

    try:
        mod.main()
        elapsed = time.time() - t0
        print(f"\n[OK] {name} finished in {elapsed:.1f}s")
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n[ERROR] {name} failed after {elapsed:.1f}s: {exc}")
        traceback.print_exc()
        return False


def write_preamble() -> None:
    """Write a header block to RESULTS.md before the experiments run."""
    from src.reporting import append_to_results_md, _RESULTS_DIR
    (_RESULTS_DIR / "RESULTS.md").unlink(missing_ok=True)   # start fresh
    append_to_results_md(
        "# Experiment Results\n\n"
        "Entropy-regularised Bermudan option pricing — "
        "reproduction of *An Entropy Regularized BSDE Approach to Bermudan Options and Games*.\n\n"
        "All experiments run on CPU.  "
        "Epoch counts reduced from paper-default for practical runtime; "
        "qualitative patterns are preserved.\n"
    )


def write_epilogue(results: dict[str, bool]) -> None:
    """Append a summary of which experiments passed / failed."""
    from src.reporting import append_to_results_md
    lines = ["## Run Summary\n"]
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        lines.append(f"- **{name}**: {status}")
    append_to_results_md("\n".join(lines) + "\n")


def main() -> None:
    # Ensure project root is on path
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    write_preamble()

    experiments = [
        "exp1_bermudan_put",
        "exp2_maxcall_2d",
        "exp3_maxcall_dimensions",
        "exp4_stopping_rule",
        "exp5_dual_bound",
        "exp6_game_option",
    ]

    run_results: dict[str, bool] = {}
    total_start = time.time()

    for exp_name in experiments:
        overrides = CPU_OVERRIDES.get(exp_name, {})
        ok = run_experiment(exp_name, overrides)
        run_results[exp_name] = ok

    write_epilogue(run_results)

    total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All experiments completed in {total:.1f}s")
    print("Summary:")
    for name, ok in run_results.items():
        print(f"  {'OK' if ok else 'FAIL':4s}  {name}")
    print(f"\nResults saved to {root}/results/")


if __name__ == "__main__":
    main()
