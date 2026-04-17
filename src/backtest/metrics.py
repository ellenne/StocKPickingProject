"""
src/backtest/metrics.py
───────────────────────
Performance metric calculations for backtest results.

Metrics computed
────────────────
  - Annualised return (CAGR)
  - Annualised volatility
  - Sharpe ratio (annualised, risk-free rate = 0 per paper)
  - Maximum drawdown
  - Turnover (annualised, one-sided)
  - Hit rate (classification accuracy on the test set)
  - Top-decile accuracy
  - CAPM alpha & beta
  - Rolling 12M metrics
  - Confusion matrix stats
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

WEEKS_PER_YEAR = 52


def _safe_div(a: float, b: float, default: float = float("nan")) -> float:
    return a / b if b != 0 else default


def annualised_return(returns: pd.Series) -> float:
    """Compound annualised return from weekly log-returns."""
    total = returns.sum()
    n_weeks = len(returns)
    return float(np.exp(total * WEEKS_PER_YEAR / n_weeks) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of weekly log-returns."""
    return float(returns.std() * np.sqrt(WEEKS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio (risk-free rate in weekly terms)."""
    excess = returns - risk_free / WEEKS_PER_YEAR
    vol = excess.std()
    return float(_safe_div(excess.mean() * WEEKS_PER_YEAR, vol * np.sqrt(WEEKS_PER_YEAR)))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def capm_alpha(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
) -> tuple[float, float]:
    """Compute CAPM alpha and beta via OLS.

    Returns (alpha_annualised, beta).
    """
    aligned = pd.concat(
        [strategy_returns.rename("strat"), market_returns.rename("mkt")], axis=1
    ).dropna()
    if len(aligned) < 10:
        return float("nan"), float("nan")
    slope, intercept, *_ = stats.linregress(aligned["mkt"], aligned["strat"])
    alpha_ann = float(intercept * WEEKS_PER_YEAR)
    return alpha_ann, float(slope)


def hit_rate(predictions: pd.DataFrame, prob_col: str, target_col: str = "target") -> float:
    """Classification accuracy: fraction of correctly predicted directions."""
    df = predictions[[prob_col, target_col]].dropna()
    pred_label = (df[prob_col] >= 0.5).astype(int)
    return float((pred_label == df[target_col]).mean())


def top_quantile_accuracy(
    predictions: pd.DataFrame,
    prob_col: str,
    quantile: float = 0.10,
    target_col: str = "target",
) -> float:
    """Accuracy in the top (and bottom) *quantile* of predicted probabilities.

    Mirrors Table 2 of the paper (10 %, 5 %, 1 % accuracy).
    """
    df = predictions[[prob_col, target_col]].dropna()
    n = len(df)
    k = max(1, int(np.ceil(n * quantile)))

    top_k = df.nlargest(k, prob_col)
    acc_high = float((top_k[target_col] == 1).mean())

    bot_k = df.nsmallest(k, prob_col)
    acc_low = float((bot_k[target_col] == 0).mean())

    return (acc_high + acc_low) / 2


def rolling_sharpe(returns: pd.Series, window: int = WEEKS_PER_YEAR) -> pd.Series:
    """Compute rolling 12M (52-week) Sharpe ratio."""
    def _roll_sharpe(r):
        if len(r) < 4:
            return float("nan")
        return sharpe_ratio(r)

    return returns.rolling(window).apply(_roll_sharpe, raw=False)


def compute_all_metrics(
    strategy_returns: pd.Series,
    benchmark_1n_returns: pd.Series,
    spy_returns: pd.Series,
    predictions: pd.DataFrame,
    prob_col: str,
    holdings: pd.DataFrame,
    one_way_bps: float = 5.0,
    tc_enabled: bool = True,
) -> dict:
    """Compute the full metrics table for one model / strategy.

    Returns
    -------
    dict
        Flat dict of metric_name → value.
    """
    from src.backtest.transaction_costs import (
        apply_transaction_costs,
        compute_btc,
        compute_turnover,
    )

    # Align all return series to the strategy dates
    idx = strategy_returns.index
    bench = benchmark_1n_returns.reindex(idx).fillna(0)
    spy = spy_returns.reindex(idx).fillna(0)

    turnover_series = compute_turnover(holdings.reindex(idx).fillna(0))
    ann_turnover = float(turnover_series.mean() * WEEKS_PER_YEAR)

    # Net returns
    if tc_enabled and len(holdings) > 0:
        net_returns = apply_transaction_costs(strategy_returns, holdings, one_way_bps)
    else:
        net_returns = strategy_returns

    capm_a, capm_b = capm_alpha(strategy_returns, spy)
    btc = compute_btc(strategy_returns, bench, holdings.reindex(idx).fillna(0))

    m = {
        "ann_return_gross": annualised_return(strategy_returns),
        "ann_return_net": annualised_return(net_returns),
        "ann_volatility": annualised_volatility(strategy_returns),
        "sharpe_gross": sharpe_ratio(strategy_returns),
        "sharpe_net": sharpe_ratio(net_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "ann_turnover": ann_turnover,
        "capm_alpha": capm_a,
        "capm_beta": capm_b,
        "hit_rate": hit_rate(predictions, prob_col),
        "top_10pct_accuracy": top_quantile_accuracy(predictions, prob_col, 0.10),
        "top_5pct_accuracy": top_quantile_accuracy(predictions, prob_col, 0.05),
        "top_1pct_accuracy": top_quantile_accuracy(predictions, prob_col, 0.01),
        "btc_bps": btc,
        # Benchmark for comparison
        "bench_1n_ann_return": annualised_return(bench),
        "bench_1n_sharpe": sharpe_ratio(bench),
        "spy_ann_return": annualised_return(spy),
        "spy_sharpe": sharpe_ratio(spy),
    }
    return m


def build_metrics_table(
    all_metrics: dict[str, dict],
) -> pd.DataFrame:
    """Stack per-model metric dicts into a comparison DataFrame.

    Parameters
    ----------
    all_metrics:
        {model_name → metrics_dict} from ``compute_all_metrics``.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metrics.
    """
    rows = []
    for model_name, m in all_metrics.items():
        row = {"model": model_name, **m}
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    return df
