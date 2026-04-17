"""
src/backtest/transaction_costs.py
──────────────────────────────────
Transaction cost and turnover modelling.

The paper computes "break-even transaction costs" (BTC) – the one-way cost
in basis points at which the excess return over the 1/N benchmark is zero.
We implement:
  - One-way transaction cost deduction per rebalance
  - Portfolio turnover calculation (one-sided)
  - Net return series after costs
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_turnover(
    holdings_history: pd.DataFrame,
) -> pd.Series:
    """Compute weekly one-sided portfolio turnover.

    Parameters
    ----------
    holdings_history:
        DataFrame indexed by date, columns = tickers, values = portfolio weights
        (should sum to 1.0 for each row).

    Returns
    -------
    pd.Series
        Weekly one-sided turnover (fraction of portfolio traded each week).
        Annualised turnover = weekly turnover × 52.
    """
    # Turnover = 0.5 * sum(|w_t - w_{t-1}|) per week
    weight_change = holdings_history.diff().abs().sum(axis=1) / 2
    weight_change.iloc[0] = holdings_history.iloc[0].abs().sum() / 2  # first week
    return weight_change


def apply_transaction_costs(
    gross_returns: pd.Series,
    holdings_history: pd.DataFrame,
    one_way_bps: float = 5.0,
) -> pd.Series:
    """Deduct transaction costs from gross weekly portfolio returns.

    Parameters
    ----------
    gross_returns:
        Weekly gross log-returns of the portfolio.
    holdings_history:
        (date × ticker) weight matrix aligned to gross_returns.
    one_way_bps:
        One-way transaction cost in basis points (default = 5 bps).

    Returns
    -------
    pd.Series
        Net weekly returns after transaction costs.
    """
    one_way = one_way_bps / 10_000  # convert to fraction
    turnover = compute_turnover(holdings_history).reindex(gross_returns.index).fillna(0)
    # Cost = one_way × turnover (two-way = 2× one-way, but turnover is one-sided)
    cost = turnover * one_way * 2  # round-trip = 2 × one_way per unit of turnover
    net_returns = gross_returns - cost
    logger.info(
        "Transaction costs applied: avg weekly cost = %.2f bps",
        cost.mean() * 10_000,
    )
    return net_returns


def compute_btc(
    strategy_gross_returns: pd.Series,
    benchmark_returns: pd.Series,
    holdings_history: pd.DataFrame,
) -> float:
    """Break-even transaction costs (BTC) in basis points.

    BTC is the one-way cost level at which the strategy's excess return over
    the benchmark equals zero.

    Returns
    -------
    float
        One-way BTC in basis points.
    """
    excess = (strategy_gross_returns - benchmark_returns).mean()
    turnover = compute_turnover(holdings_history).reindex(strategy_gross_returns.index).fillna(0)
    avg_turnover = turnover.mean()  # weekly turnover fraction

    if avg_turnover == 0:
        return float("inf")

    # excess = BTC × avg_turnover × 2 (round-trip)  →  BTC = excess / (2 × turnover)
    btc_bps = (excess / (2 * avg_turnover)) * 10_000
    return float(btc_bps)
