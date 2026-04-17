"""
tests/test_features.py
─────────────────────
Unit tests for the feature engineering pipeline.
Tests run on synthetic data so no network access is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.technical import compute_technical_features
from src.features.preprocessing import FeaturePreprocessor, build_feature_matrix


def _make_close_panel(n_weeks: int = 120, n_tickers: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic weekly close price panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-03", periods=n_weeks, freq="W-WED")
    prices = np.exp(
        np.cumsum(rng.normal(0.002, 0.03, size=(n_weeks, n_tickers)), axis=0)
    ) * 100
    return pd.DataFrame(
        prices,
        index=dates,
        columns=[f"T{i:03d}" for i in range(n_tickers)],
    )


def _make_volume_panel(close_panel: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = rng.integers(100_000, 10_000_000, size=close_panel.shape).astype(float)
    return pd.DataFrame(vol, index=close_panel.index, columns=close_panel.columns)


# ─────────────────────────────────────────────────────────────────────────────

class TestTechnicalFeatures:
    def setup_method(self):
        self.close = _make_close_panel()
        self.volume = _make_volume_panel(self.close)
        # Synthetic SPY returns
        rng = np.random.default_rng(99)
        self.spy_ret = pd.Series(
            rng.normal(0.002, 0.02, len(self.close)),
            index=self.close.index,
            name="SPY",
        )

    def test_output_shape(self):
        features = compute_technical_features(self.close, self.volume, self.spy_ret)
        n_stocks = len(self.close.columns)
        n_weeks = len(self.close)
        assert len(features) == n_stocks * n_weeks
        assert features.index.names == ["date", "ticker"]

    def test_expected_columns_present(self):
        features = compute_technical_features(self.close, self.volume, self.spy_ret)
        expected = [
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
        for col in expected:
            assert col in features.columns, f"Missing column: {col}"

    def test_no_lookahead_momentum(self):
        """Momentum at time t should be based only on closes up to t-1."""
        features = compute_technical_features(self.close, self.volume)
        # mom_1m at the earliest date should be NaN (not enough history)
        earliest = features.index.get_level_values("date").min()
        early_vals = features.xs(earliest, level="date")["mom_1m"]
        assert early_vals.isna().all(), "Momentum should be NaN at start"

    def test_rsi_bounded(self):
        features = compute_technical_features(self.close, self.volume)
        rsi_vals = features["rsi_14"].dropna()
        assert (rsi_vals >= 0).all() and (rsi_vals <= 100).all()

    def test_without_spy_skips_relative_momentum(self):
        features = compute_technical_features(self.close, self.volume, None)
        assert "rel_mom_12m" not in features.columns

    def test_usd_volume_positive(self):
        features = compute_technical_features(self.close, self.volume)
        usd_vol = features["usd_volume"].dropna()
        # log(price * volume) – values should be positive logs (prices > 0)
        assert (usd_vol > 0).mean() > 0.9


class TestFeaturePreprocessor:
    def setup_method(self):
        n_samples, n_features = 200, 15
        rng = np.random.default_rng(42)
        dates = pd.date_range("2018-01-01", periods=n_samples, freq="W")
        self.X = pd.DataFrame(
            rng.normal(0, 1, (n_samples, n_features)),
            index=pd.MultiIndex.from_arrays(
                [dates, [f"T{i}" for i in range(n_samples)]],
                names=["date", "ticker"],
            ),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        # Inject some outliers
        self.X.iloc[0, 0] = 1000.0
        self.X.iloc[-1, -1] = -1000.0

    def test_fit_transform_shape(self):
        prep = FeaturePreprocessor(winsorize=True, winsorize_pct=0.01)
        prep.fit(self.X)
        X_scaled = prep.transform(self.X)
        assert X_scaled.shape == self.X.shape

    def test_standardised_mean_near_zero(self):
        prep = FeaturePreprocessor(winsorize=True, winsorize_pct=0.01)
        prep.fit(self.X)
        X_scaled = prep.transform(self.X)
        # After standardisation means should be close to 0
        assert (X_scaled.mean().abs() < 0.5).all()

    def test_winsorisation_removes_extreme_values(self):
        prep = FeaturePreprocessor(winsorize=True, winsorize_pct=0.01)
        prep.fit(self.X)
        X_scaled = prep.transform(self.X)
        # Standardised outlier should no longer be extreme
        assert X_scaled.abs().max().max() < 50  # was 1000 before

    def test_no_nan_after_transform(self):
        prep = FeaturePreprocessor(winsorize=True)
        prep.fit(self.X.fillna(0))
        X_scaled = prep.transform(self.X.fillna(0))
        assert not X_scaled.isna().any().any()

    def test_train_only_statistics(self):
        """Statistics computed on training set only – transform does not refit."""
        half = len(self.X) // 2
        X_train = self.X.iloc[:half]
        X_test = self.X.iloc[half:]
        prep = FeaturePreprocessor(winsorize=False)
        prep.fit(X_train)
        # Means stored from train set, not updated on test
        train_means = prep._means.copy()
        prep.transform(X_test)
        assert (prep._means == train_means).all()
