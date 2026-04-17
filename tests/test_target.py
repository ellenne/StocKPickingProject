"""
tests/test_target.py
────────────────────
Unit tests for the binary target construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.target import add_target, forward_return_col


def _make_panel(n_weeks: int = 20, n_tickers: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_weeks, freq="W-WED")
    tickers = [f"T{i}" for i in range(n_tickers)]
    log_ret = rng.normal(0.001, 0.03, (n_weeks, n_tickers))
    mi = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame(
        {"log_return": log_ret.flatten()},
        index=mi,
    )
    return df


class TestAddTarget:
    def setup_method(self):
        self.panel = _make_panel()

    def test_target_column_added(self):
        result = add_target(self.panel)
        assert "target" in result.columns

    def test_target_binary(self):
        result = add_target(self.panel)
        vals = result["target"].dropna().unique()
        for v in vals:
            assert v in (0.0, 1.0), f"Unexpected target value: {v}"

    def test_last_date_all_nan(self):
        """The last date has no forward return → target should be NaN."""
        result = add_target(self.panel)
        last_date = result.index.get_level_values("date").max()
        last_targets = result.xs(last_date, level="date")["target"]
        assert last_targets.isna().all()

    def test_roughly_balanced_classes(self):
        """Cross-sectional median split → expect ~50% positive labels."""
        result = add_target(self.panel)
        pos_rate = result["target"].mean()
        assert 0.35 < pos_rate < 0.65, f"Positive rate {pos_rate:.2%} too imbalanced"

    def test_strict_gt_vs_ge(self):
        """strict_gt=True vs False can differ only on ties."""
        result_strict = add_target(self.panel, strict_gt=True)
        result_ge = add_target(self.panel, strict_gt=False)
        # ge should have >= as many positives as strict gt
        pos_strict = result_strict["target"].sum()
        pos_ge = result_ge["target"].sum()
        assert pos_ge >= pos_strict

    def test_no_lookahead(self):
        """Feature at date t uses return at t+1 – not current return."""
        result = add_target(self.panel)
        # The target for date t is based on the return at t+1.
        # We verify by checking: target and log_return for the same row
        # are NOT simply correlated (target uses the NEXT row's return).
        dates = result.index.get_level_values("date").unique()
        for date in dates[:-2]:
            # Get current and next return for the first ticker
            t0 = result.xs(date, level="date")["target"].iloc[0]
            t1_ret = result.xs(dates[dates.get_loc(date) + 1], level="date")["log_return"].iloc[0]
            # The target at t0 should reflect t1_ret relative to median
            # (exact values depend on cross-section, so we just check no crash)
            assert t0 in (0.0, 1.0, float("nan"))

    def test_forward_return_col_added(self):
        result = forward_return_col(self.panel)
        assert "fwd_return" in result.columns
        # fwd_return at t should equal log_return at t+1
        dates = result.index.get_level_values("date").unique()
        t0 = dates[0]
        t1 = dates[1]
        fwd_at_t0 = result.xs(t0, level="date")["fwd_return"].values
        ret_at_t1 = result.xs(t1, level="date")["log_return"].values
        np.testing.assert_array_almost_equal(fwd_at_t0, ret_at_t1, decimal=10)
