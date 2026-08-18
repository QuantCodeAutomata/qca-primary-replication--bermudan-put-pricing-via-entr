"""
Result table generation, formatting, and persistence utilities for the
entropy-regularised Bermudan option pricing experiments.

Outputs are written to the ``results/`` directory at the repository root.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# Path to the project-level results directory
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _ensure_results_dir() -> None:
    """Create the results directory if it does not already exist."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_policy_iteration_table(
    results: dict,
    td_results: dict,
    n_iters: int,
) -> pd.DataFrame:
    """
    Build a summary DataFrame in the style of Tables 1/2 from the paper.

    Each row corresponds to one value of the regularisation parameter λ.
    Columns show the option price after each policy-iteration step and the
    final TD-learning price.

    Parameters
    ----------
    results : dict
        Mapping ``{lambda_value: [price_iter_0, price_iter_1, ...]}``.
        Lists may be shorter than ``n_iters``; missing entries are filled
        with ``NaN``.
    td_results : dict
        Mapping ``{lambda_value: td_price}`` for the TD-learning estimate.
        Missing keys produce ``NaN`` in the TD column.
    n_iters : int
        Number of policy-iteration columns to include in the table.

    Returns
    -------
    pd.DataFrame
        Index: λ values (sorted ascending).
        Columns: ``"Iter 0"``, ``"Iter 1"``, …, ``"Iter {n_iters-1}"``,
        then ``"TD"``.

    Examples
    --------
    >>> results   = {0.1: [3.8, 3.9, 4.0], 0.5: [3.5, 3.85, 3.95]}
    >>> td_results = {0.1: 4.01, 0.5: 3.96}
    >>> df = make_policy_iteration_table(results, td_results, n_iters=3)
    """
    iter_cols = [f"Iter {i}" for i in range(n_iters)]
    rows = []
    for lam in sorted(results.keys()):
        prices = results[lam]
        row: dict = {"lambda": lam}
        for i, col in enumerate(iter_cols):
            row[col] = prices[i] if i < len(prices) else np.nan
        row["TD"] = td_results.get(lam, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("lambda")
    df.index.name = "λ"
    return df


def save_table_to_results(
    df: pd.DataFrame,
    filename: str,
    title: str,
) -> None:
    """
    Persist a result DataFrame as both CSV and a formatted text table.

    Files are written to the ``results/`` directory:

    * ``results/{filename}.csv``  — machine-readable CSV.
    * ``results/{filename}.txt``  — human-readable table with title header.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save (typically from :func:`make_policy_iteration_table`).
    filename : str
        Base name without extension, e.g. ``"table1_bermudan_put"``.
    title : str
        Descriptive title printed at the top of the text file.

    Returns
    -------
    None
    """
    _ensure_results_dir()

    csv_path = _RESULTS_DIR / f"{filename}.csv"
    txt_path = _RESULTS_DIR / f"{filename}.txt"

    df.to_csv(csv_path)

    separator = "=" * max(len(title), 60)
    text_block = "\n".join([
        separator,
        title,
        separator,
        df.to_string(float_format="{:.4f}".format),
        "",
    ])
    txt_path.write_text(text_block, encoding="utf-8")

    print(f"[reporting] Saved {csv_path.name} and {txt_path.name} to {_RESULTS_DIR}")


def append_to_results_md(content: str) -> None:
    """
    Append a string block to ``results/RESULTS.md``.

    The file is created if it does not exist.  A newline is always
    inserted between the existing content and the new block.

    Parameters
    ----------
    content : str
        Markdown-formatted text to append.

    Returns
    -------
    None
    """
    _ensure_results_dir()
    md_path = _RESULTS_DIR / "RESULTS.md"

    with md_path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(content)
        fh.write("\n")
