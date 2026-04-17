"""
src/backtest/portfolio.py
─────────────────────────
Portfolio construction: select top-N stocks by predicted outperformance
probability, hold equally weighted, compute weekly returns.

Design
──────
* Equal-weight long-only portfolio: each selected stock gets weight 1/N.
* Benchmark 1: equal-weight over the full universe (1/N benchmark).
* Benchmark 2: SPY weekly returns (S&P 500 total-return proxy).
* Returns are computed as simple arithmetic returns for the equity curve,
  but the raw data is log-returns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def select_top_n(
    proba_series: pd.Series,
    n: int,
) -> list[str]:
    """Return the tickers with the top-N outperformance probabilities.

    Parameters
    ----------
    proba_series:
        Index = ticker, values = P(outperform).  Should be cross-sectional
        for a single rebalance date.
    n:
        Number of stocks to select.

    Returns
    -------
    list[str]
    """
    valid = proba_series.dropna()
    n = min(n, len(valid))
    return valid.nlargest(n).index.tolist()


def build_portfolio_returns(
    predictions: pd.DataFrame,
    fwd_return_col: str = "fwd_return",
    prob_col: str = "ensemble_prob",
    top_n: int = 50,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute weekly portfolio returns and holdings history.

    Parameters
    ----------
    predictions:
        MultiIndex (date, ticker) predictions DataFrame.  Must contain
        ``fwd_return_col`` and ``prob_col``.
    fwd_return_col:
        Column with the next-week log-return.
    prob_col:
        Column with the predicted outperformance probability.
    top_n:
        Number of stocks to hold each week.

    Returns
    -------
    portfolio_returns: pd.Series
        Weekly log-returns of the equal-weight portfolio.
    holdings: pd.DataFrame
        (date × ticker) weight matrix (0 or 1/N for selected stocks).
    """
    if prob_col not in predictions.columns:
        raise ValueError(f"Column '{prob_col}' not found in predictions.")
    if fwd_return_col not in predictions.columns:
        raise ValueError(f"Column '{fwd_return_col}' not found in predictions.")

    dates = predictions.index.get_level_values("date").unique().sort_values()
    all_tickers = predictions.index.get_level_values("ticker").unique().tolist()

    port_returns: list[float] = []
    port_dates: list[pd.Timestamp] = []
    holdings_rows: list[dict] = []

    for date in dates:
        day_pred = predictions.xs(date, level="date")

        proba = day_pred[prob_col].dropna()
        fwd_ret = day_pred[fwd_return_col]

        if proba.empty:
            continue

        selected = select_top_n(proba, top_n)
        weight = 1.0 / len(selected)

        # Portfolio return = equal-weight average of forward returns
        ret_vals = fwd_ret.reindex(selected).dropna()
        if ret_vals.empty:
            continue

        port_ret = ret_vals.mean()  # log-return average (approx equal-weight)
        port_returns.append(port_ret)
        port_dates.append(date)

        # Build holdings row
        row = {t: weight if t in selected else 0.0 for t in all_tickers}
        row["date"] = date
        holdings_rows.append(row)

    portfolio_returns = pd.Series(port_returns, index=pd.DatetimeIndex(port_dates), name="portfolio")
    holdings = pd.DataFrame(holdings_rows).set_index("date").fillna(0.0)

    logger.info(
        "Portfolio built: %d weeks, avg held=%.1f stocks",
        len(portfolio_returns),
        (holdings > 0).sum(axis=1).mean(),
    )
    return portfolio_returns, holdings


def build_benchmark_returns(
    predictions: pd.DataFrame,
    fwd_return_col: str = "fwd_return",
) -> pd.Series:
    """Compute the equal-weight 1/N benchmark returns over the universe.

    Returns the cross-sectional mean forward return for each date in the
    predictions panel.
    """
    ret = predictions[fwd_return_col].unstack("ticker")
    benchmark = ret.mean(axis=1)
    benchmark.name = "benchmark_1n"
    return benchmark


def download_spy_returns(
    start: str,
    end: str,
    cache_dir,
    force_refresh: bool = False,
) -> pd.Series:
    """Download weekly SPY returns as the market benchmark.

    Returns
    -------
    pd.Series
        Weekly log-returns indexed by Wednesday dates.
    """
    from src.data.caching import cached_download
    from src.data.prices import _daily_to_weekly

    def _fetch():
        raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.Series(dtype=float, name="SPY").to_frame()
        # Handle MultiIndex columns from yfinance >= 1.3.0
        if isinstance(raw.columns, pd.MultiIndex):
            ticker_level = raw.columns.names.index("Ticker") if "Ticker" in raw.columns.names else 1
            raw = raw.xs("SPY", axis=1, level=ticker_level)
        weekly = _daily_to_weekly(raw)
        log_ret = np.log(weekly["Close"] / weekly["Close"].shift(1))
        return log_ret.rename("SPY").to_frame()

    df = cached_download("spy_returns", _fetch, cache_dir, force_refresh)
    if isinstance(df, pd.DataFrame) and "SPY" in df.columns:
        return df["SPY"]
    return pd.Series(dtype=float, name="SPY")
