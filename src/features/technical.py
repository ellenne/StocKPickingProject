"""
src/features/technical.py
─────────────────────────
Computes all technical indicator features described in Table 1 (Panel B)
of Wolff & Echterling (2022).

All calculations are performed ticker-by-ticker on daily-frequency adjusted
close prices (already resampled to weekly frequency in the price panel).

Weekly convention: a week's features are calculated from the *close* price
on that Wednesday relative to historical closes up to that Wednesday.

Feature list (paper-faithful)
──────────────────────────────
Momentum:
  mom_12m, mom_6m, mom_1m
  rel_mom_12m, rel_mom_6m, rel_mom_1m   (vs S&P 500 proxy = SPY)

Moving averages:
  log_price_ma200, log_price_ma100, log_price_ma50

Risk:
  beta_12m
  vol_12m, vol_6m, vol_1m

Short-term reversal (RSI):
  rsi_14, rsi_9, rsi_3
  log_price_bb_upper, log_price_bb_lower

Lagged returns:
  ret_lag1, ret_lag2

Volume:
  usd_volume   (price × volume)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate trading days per week / year at weekly frequency
_WEEKS_PER_YEAR = 52
_WEEKS_12M = 52
_WEEKS_6M = 26
_WEEKS_1M = 4

# Moving average windows are in *days* – we approximate with weekly equivalents
# 200 trading days ≈ 40 weeks, 100d ≈ 20 weeks, 50d ≈ 10 weeks
_MA200_WEEKS = 40
_MA100_WEEKS = 20
_MA50_WEEKS = 10

# Bollinger band window and sigma (standard in literature)
_BB_WINDOW = 20   # trading days; at weekly freq ≈ 4 weeks
_BB_WEEKS = 4
_BB_STD = 2.0


def _rsi(series: pd.Series, window: int) -> pd.Series:
    """Wilder RSI.  ``window`` is in the same units as ``series`` index."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_technical_features(
    close_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    index_return_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute all technical features from weekly close and volume panels.

    Parameters
    ----------
    close_panel:
        (n_weeks × n_tickers) weekly adjusted close prices.
    volume_panel:
        (n_weeks × n_tickers) weekly trading volume.
    index_return_series:
        Weekly log-return of the benchmark index (SPY).  If None, relative
        momentum features are omitted.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker) with one column per technical feature.
    """
    n_weeks, n_tickers = close_panel.shape
    log_ret = np.log(close_panel / close_panel.shift(1))

    features: dict[str, pd.DataFrame] = {}

    # ── Momentum ─────────────────────────────────────────────────────────
    # Momentum = cumulative log-return over past window (skip last week)
    # Following standard convention: skip the most recent week (t-1 skip)
    features["mom_12m"] = log_ret.rolling(_WEEKS_12M).sum().shift(1)
    features["mom_6m"] = log_ret.rolling(_WEEKS_6M).sum().shift(1)
    features["mom_1m"] = log_ret.rolling(_WEEKS_1M).sum().shift(1)

    if index_return_series is not None:
        idx = index_return_series.reindex(close_panel.index).fillna(0)
        idx_12m = idx.rolling(_WEEKS_12M).sum().shift(1)
        idx_6m = idx.rolling(_WEEKS_6M).sum().shift(1)
        idx_1m = idx.rolling(_WEEKS_1M).sum().shift(1)
        features["rel_mom_12m"] = features["mom_12m"].sub(idx_12m, axis=0)
        features["rel_mom_6m"] = features["mom_6m"].sub(idx_6m, axis=0)
        features["rel_mom_1m"] = features["mom_1m"].sub(idx_1m, axis=0)

    # ── Moving averages ───────────────────────────────────────────────────
    ma200 = close_panel.rolling(_MA200_WEEKS, min_periods=_MA200_WEEKS // 2).mean()
    ma100 = close_panel.rolling(_MA100_WEEKS, min_periods=_MA100_WEEKS // 2).mean()
    ma50 = close_panel.rolling(_MA50_WEEKS, min_periods=_MA50_WEEKS // 2).mean()

    features["log_price_ma200"] = np.log(close_panel / ma200.replace(0, np.nan))
    features["log_price_ma100"] = np.log(close_panel / ma100.replace(0, np.nan))
    features["log_price_ma50"] = np.log(close_panel / ma50.replace(0, np.nan))

    # ── Beta 12M ──────────────────────────────────────────────────────────
    if index_return_series is not None:
        idx_r = index_return_series.reindex(close_panel.index).fillna(0)
        beta_dict = {}
        for col in close_panel.columns:
            stk_r = log_ret[col]
            cov = stk_r.rolling(_WEEKS_12M).cov(idx_r)
            var = idx_r.rolling(_WEEKS_12M).var()
            beta_dict[col] = cov / var.replace(0, np.nan)
        features["beta_12m"] = pd.DataFrame(beta_dict, index=close_panel.index)

    # ── Volatility ────────────────────────────────────────────────────────
    features["vol_12m"] = log_ret.rolling(_WEEKS_12M).std() * np.sqrt(_WEEKS_PER_YEAR)
    features["vol_6m"] = log_ret.rolling(_WEEKS_6M).std() * np.sqrt(_WEEKS_PER_YEAR)
    features["vol_1m"] = log_ret.rolling(_WEEKS_1M).std() * np.sqrt(_WEEKS_PER_YEAR)

    # ── RSI ───────────────────────────────────────────────────────────────
    # Paper uses daily RSI(14), RSI(9), RSI(3); we approximate at weekly freq.
    # Mapping: 14d ≈ 3 weeks, 9d ≈ 2 weeks, 3d ≈ 2 weeks (minimum viable).
    # Note: window=1 (for 3d→1w) gives Wilder alpha=1.0 (zero memory), which
    # produces ~55% NaN (undefined RSI when loss_ewm==0).  Using window=2 as
    # the minimum meaningful Wilder window at weekly frequency.
    rsi_dict_14 = {col: _rsi(close_panel[col], 3) for col in close_panel.columns}
    rsi_dict_9 = {col: _rsi(close_panel[col], 2) for col in close_panel.columns}
    rsi_dict_3 = {col: _rsi(close_panel[col], 2) for col in close_panel.columns}  # window≥2
    features["rsi_14"] = pd.DataFrame(rsi_dict_14, index=close_panel.index)
    features["rsi_9"] = pd.DataFrame(rsi_dict_9, index=close_panel.index)
    features["rsi_3"] = pd.DataFrame(rsi_dict_3, index=close_panel.index)

    # ── Bollinger bands ───────────────────────────────────────────────────
    bb_ma = close_panel.rolling(_BB_WEEKS, min_periods=2).mean()
    bb_std = close_panel.rolling(_BB_WEEKS, min_periods=2).std()
    bb_upper = bb_ma + _BB_STD * bb_std
    bb_lower = bb_ma - _BB_STD * bb_std
    features["log_price_bb_upper"] = np.log(
        close_panel / bb_upper.replace(0, np.nan)
    )
    features["log_price_bb_lower"] = np.log(
        close_panel / bb_lower.replace(0, np.nan)
    )

    # ── Lagged returns ────────────────────────────────────────────────────
    features["ret_lag1"] = log_ret.shift(1)
    features["ret_lag2"] = log_ret.shift(2)

    # ── USD volume ────────────────────────────────────────────────────────
    features["usd_volume"] = np.log(
        (close_panel * volume_panel).replace(0, np.nan)
    )

    # ── Stack all features into long format (date, ticker) ────────────────
    stacked_frames = []
    for fname, df in features.items():
        melted = df.rename_axis("date").reset_index().melt(
            id_vars="date", var_name="ticker", value_name=fname
        )
        stacked_frames.append(melted.set_index(["date", "ticker"]))

    result = pd.concat(stacked_frames, axis=1)
    logger.info(
        "Technical features computed: %d features, panel shape %s",
        len(features),
        result.shape,
    )
    return result
