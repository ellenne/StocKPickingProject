"""
src/features/target.py
──────────────────────
Computes the binary prediction target defined in the paper:

    label_t = 1  iff  return_{t+1} > cross-sectional median return_{t+1}
              0  otherwise

The target is computed for the *next* week's return so that at each
rebalance date t the label describes whether the stock outperforms in the
subsequent week [t+1].

To avoid lookahead bias:
  - The return used for labelling is the *forward* return (next period).
  - The cross-sectional median is computed *across all stocks at date t+1*,
    not using any future information beyond that single weekly return.
  - Features are computed from data up to and including date t.

Usage
─────
    from src.features.target import add_target
    panel = add_target(panel, strict_gt=True)
    # panel now has a 'target' column (int 0/1)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_target(
    panel: pd.DataFrame,
    return_col: str = "log_return",
    strict_gt: bool = True,
) -> pd.DataFrame:
    """Compute and attach the binary target column to *panel*.

    Parameters
    ----------
    panel:
        MultiIndex (date, ticker) panel with a log-return column.
    return_col:
        Name of the weekly log-return column.
    strict_gt:
        If ``True`` (paper default), label = 1 iff return > median (strict).
        If ``False``, label = 1 iff return >= median.

    Returns
    -------
    pd.DataFrame
        Input panel with an additional ``target`` column (int 0/1, NaN on
        the last date where no forward return exists).
    """
    if return_col not in panel.columns:
        raise ValueError(f"Column '{return_col}' not found in panel.")

    panel = panel.copy()

    # Extract return as (date × ticker) wide format for convenience
    ret = panel[return_col].unstack("ticker")  # shape (n_dates, n_tickers)

    # Forward return: return at date t+1 seen from date t
    fwd_ret = ret.shift(-1)  # rows t contain return that materialises at t+1

    # Cross-sectional median at each date (computed on the *forward* date)
    xs_median = fwd_ret.median(axis=1)  # shape (n_dates,)

    # Binary label
    if strict_gt:
        label = fwd_ret.gt(xs_median, axis=0).astype(float)
    else:
        label = fwd_ret.ge(xs_median, axis=0).astype(float)

    # NaN where forward return is unavailable (last date)
    label[fwd_ret.isna()] = float("nan")

    # Stack back to long format and merge
    label_long = label.stack(future_stack=True).rename("target")
    label_long = label_long.reindex(panel.index)

    panel["target"] = label_long.values

    n_total = panel["target"].notna().sum()
    n_pos = (panel["target"] == 1).sum()
    logger.info(
        "Target computed: %d labelled rows, %.1f%% positive (expect ~50%%)",
        n_total,
        100 * n_pos / max(n_total, 1),
    )
    return panel


def forward_return_col(
    panel: pd.DataFrame,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Add a ``fwd_return`` column (next-week return) to the panel.

    Useful for computing portfolio returns after the fact.
    """
    ret = panel[return_col].unstack("ticker")
    fwd = ret.shift(-1).stack(future_stack=True).rename("fwd_return")
    fwd = fwd.reindex(panel.index)
    panel = panel.copy()
    panel["fwd_return"] = fwd.values
    return panel
