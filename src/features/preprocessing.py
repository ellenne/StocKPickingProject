"""
src/features/preprocessing.py
──────────────────────────────
Feature preprocessing pipeline faithful to the paper:

1. Forward-fill missing values within each ticker (up to *ffill_limit* weeks).
2. Cross-sectional median imputation for remaining NaNs (by date).
3. Log-transform market cap and USD volume to reduce skewness.
4. Optional winsorisation at the p-th percentile.
5. Standardise using training-set mean/std (no future leakage).

The ``FeaturePreprocessor`` follows the scikit-learn ``fit / transform``
interface so it can be rebuilt cleanly at each training window.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.fundamentals import (
    FUNDAMENTAL_FEATURE_COLS,
    get_fundamental_feature_cols,
)

logger = logging.getLogger(__name__)

# Technical feature columns (must match names produced by technical.py)
TECHNICAL_FEATURE_COLS: list[str] = [
    "mom_12m", "mom_6m", "mom_1m",
    "rel_mom_12m", "rel_mom_6m", "rel_mom_1m",
    "log_price_ma200", "log_price_ma100", "log_price_ma50",
    "beta_12m",
    "vol_12m", "vol_6m", "vol_1m",
    "rsi_14", "rsi_9", "rsi_3",
    "log_price_bb_upper", "log_price_bb_lower",
    "ret_lag1", "ret_lag2",
    "usd_volume",
]

# Columns to log-transform before standardisation
_LOG_TRANSFORM_COLS: frozenset[str] = frozenset(["market_cap"])


def build_feature_matrix(
    panel: pd.DataFrame,
    ffill_limit: int = 52,
    include_technical: bool = True,
    include_fundamental: bool = True,
    include_sector_dummies: bool = True,
) -> pd.DataFrame:
    """Apply forward-fill and cross-sectional median imputation.

    This step is applied to the *whole* panel before any train/test split so
    that forward-filling uses temporal ordering correctly.  The standardisation
    step (fit on training set only) is handled by ``FeaturePreprocessor``.

    Parameters
    ----------
    panel:
        MultiIndex (date, ticker) panel from ``loaders.build_dataset``.
    ffill_limit:
        Maximum number of consecutive weeks to forward-fill within a ticker.
    include_*:
        Feature group toggles.

    Returns
    -------
    pd.DataFrame
        Same index as *panel*, only feature columns retained (no OHLCV, no
        log_return, no target – those are kept in the caller).
    """
    feature_cols: list[str] = []

    if include_technical:
        tech = [c for c in TECHNICAL_FEATURE_COLS if c in panel.columns]
        feature_cols.extend(tech)

    if include_fundamental:
        fund = get_fundamental_feature_cols(panel)
        feature_cols.extend(fund)

    if include_sector_dummies:
        sector_cols = [c for c in panel.columns if c.startswith("sector_")]
        feature_cols.extend(sector_cols)

    feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate, preserve order

    df = panel[feature_cols].copy()

    # Log-transform selected columns
    for col in _LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ── Step 1: forward-fill within ticker ────────────────────────────────
    # Unstack to wide format (date × ticker), ffill along date axis per column,
    # then restack.  This is ~10× faster than groupby(ticker).apply(ffill) on
    # a 277 k-row panel and produces identical results.
    ffill_parts = []
    for col in df.columns:
        wide = df[col].unstack("ticker")          # shape (n_dates, n_tickers)
        wide = wide.ffill(limit=ffill_limit)       # fill along dates axis (per ticker)
        ffill_parts.append(wide.stack(future_stack=True).rename(col))
    df = pd.concat(ffill_parts, axis=1)

    # ── Step 2: cross-sectional median imputation by date ─────────────────
    # CORRECT: transform("median") computes the median INDEPENDENTLY per date
    # (the cross-section of ~470 tickers on that specific Wednesday).
    #
    # DO NOT use df.fillna(df.median()):  df.median() is the global (all-time)
    # median which (a) leaks future values into the 3-month fundamental lag
    # window and (b) ignores market-regime variation across dates.
    xs_medians = df.groupby(level="date").transform("median")
    df = df.fillna(xs_medians)

    logger.info(
        "Feature matrix built: %d rows × %d features  (remaining NaN: %.2f%%)",
        len(df),
        len(feature_cols),
        100 * df.isna().mean().mean(),
    )
    return df


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Winsorisation + standardisation fitted on training data only.

    Parameters
    ----------
    winsorize:
        Whether to winsorise at ``winsorize_pct`` tails before standardising.
    winsorize_pct:
        Fraction for each tail (default 0.01 = 1 %).
    """

    def __init__(
        self,
        winsorize: bool = True,
        winsorize_pct: float = 0.01,
    ) -> None:
        self.winsorize = winsorize
        self.winsorize_pct = winsorize_pct
        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None
        self._lower: pd.Series | None = None
        self._upper: pd.Series | None = None
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "FeaturePreprocessor":
        """Compute winsorise bounds and z-score params from training data X."""
        self._feature_cols = X.columns.tolist()
        # Ensure all columns are float (sector dummies may be bool)
        X = X.astype(float)

        if self.winsorize:
            self._lower = X.quantile(self.winsorize_pct)
            self._upper = X.quantile(1 - self.winsorize_pct)
        else:
            self._lower = X.min()
            self._upper = X.max()

        # Compute stats *after* winsorisation on training data
        X_clipped = X.clip(lower=self._lower, upper=self._upper, axis=1)
        self._means = X_clipped.mean()
        self._stds = X_clipped.std().replace(0, 1.0)  # avoid div-by-zero
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Winsorise and z-score X using fitted statistics."""
        if self._means is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X = X.reindex(columns=self._feature_cols, fill_value=0.0).astype(float)
        X_clipped = X.clip(lower=self._lower, upper=self._upper, axis=1)
        X_scaled = (X_clipped - self._means) / self._stds
        return X_scaled.fillna(0.0)
