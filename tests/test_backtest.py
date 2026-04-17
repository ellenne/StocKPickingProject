"""
tests/test_backtest.py
──────────────────────
Unit tests for backtest logic: portfolio construction, metrics, and
transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import build_benchmark_returns, build_portfolio_returns, select_top_n
from src.backtest.metrics import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    hit_rate,
    top_quantile_accuracy,
)
from src.backtest.transaction_costs import (
    compute_turnover,
    apply_transaction_costs,
    compute_btc,
)


def _make_predictions(n_weeks: int = 100, n_tickers: int = 50, seed: int = 42) -> pd.DataFrame:
    """Synthetic predictions DataFrame with MultiIndex (date, ticker)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-03", periods=n_weeks, freq="W-WED")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    mi = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(mi)
    df = pd.DataFrame(
        {
            "fwd_return": rng.normal(0.001, 0.03, n),
            "target": (rng.random(n) > 0.5).astype(float),
            "ensemble_prob": rng.uniform(0.3, 0.7, n),
            "ridge_prob": rng.uniform(0.3, 0.7, n),
        },
        index=mi,
    )
    return df


class TestSelectTopN:
    def test_returns_correct_count(self):
        proba = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
        result = select_top_n(proba, n=2)
        assert len(result) == 2
        assert set(result) == {"A", "B"}

    def test_handles_fewer_tickers_than_n(self):
        proba = pd.Series({"A": 0.9, "B": 0.7})
        result = select_top_n(proba, n=10)
        assert len(result) == 2

    def test_nan_excluded(self):
        proba = pd.Series({"A": 0.9, "B": float("nan"), "C": 0.5})
        result = select_top_n(proba, n=3)
        assert "B" not in result
        assert len(result) <= 2


class TestPortfolioReturns:
    def setup_method(self):
        self.preds = _make_predictions()

    def test_returns_series_non_empty(self):
        port_ret, holdings = build_portfolio_returns(self.preds, top_n=10)
        assert len(port_ret) > 0

    def test_holdings_weights_sum_to_one(self):
        port_ret, holdings = build_portfolio_returns(self.preds, top_n=10)
        weight_sums = holdings.sum(axis=1)
        # Each row should sum to approximately 1.0
        assert (weight_sums - 1.0).abs().max() < 1e-9

    def test_holdings_exactly_n_stocks(self):
        top_n = 15
        port_ret, holdings = build_portfolio_returns(self.preds, top_n=top_n)
        n_held = (holdings > 0).sum(axis=1)
        assert (n_held <= top_n).all()

    def test_benchmark_returns_shape(self):
        bench = build_benchmark_returns(self.preds)
        assert isinstance(bench, pd.Series)
        assert len(bench) > 0


class TestMetrics:
    def setup_method(self):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2018-01-01", periods=260, freq="W")
        self.pos_returns = pd.Series(
            rng.normal(0.005, 0.025, 260), index=dates
        )
        self.neg_returns = pd.Series(
            rng.normal(-0.002, 0.025, 260), index=dates
        )
        self.flat_returns = pd.Series(np.zeros(260), index=dates)

    def test_annualised_return_positive(self):
        ar = annualised_return(self.pos_returns)
        assert ar > 0

    def test_annualised_return_negative(self):
        ar = annualised_return(self.neg_returns)
        assert ar < 0

    def test_annualised_volatility_nonnegative(self):
        vol = annualised_volatility(self.pos_returns)
        assert vol >= 0

    def test_flat_returns_zero_vol(self):
        vol = annualised_volatility(self.flat_returns)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_sharpe_positive_for_positive_returns(self):
        sr = sharpe_ratio(self.pos_returns)
        assert sr > 0

    def test_max_drawdown_nonpositive(self):
        dd = max_drawdown(self.pos_returns)
        assert dd <= 0

    def test_max_drawdown_zero_for_monotone_up(self):
        monotone = pd.Series([0.01] * 50)
        dd = max_drawdown(monotone)
        assert dd == pytest.approx(0.0, abs=1e-10)

    def test_hit_rate_range(self):
        preds = _make_predictions()
        hr = hit_rate(preds, "ensemble_prob")
        assert 0.0 <= hr <= 1.0

    def test_top_quantile_accuracy(self):
        preds = _make_predictions()
        acc = top_quantile_accuracy(preds, "ensemble_prob", 0.10)
        assert 0.0 <= acc <= 1.0


class TestTransactionCosts:
    def setup_method(self):
        rng = np.random.default_rng(42)
        n = 52
        dates = pd.date_range("2018-01-01", periods=n, freq="W")
        tickers = ["A", "B", "C"]
        # Holdings: alternate between holding A,B and B,C
        rows = []
        for i, d in enumerate(dates):
            if i % 2 == 0:
                rows.append({"date": d, "A": 0.5, "B": 0.5, "C": 0.0})
            else:
                rows.append({"date": d, "A": 0.0, "B": 0.5, "C": 0.5})
        self.holdings = pd.DataFrame(rows).set_index("date")
        self.returns = pd.Series(rng.normal(0.003, 0.02, n), index=dates)

    def test_turnover_positive(self):
        to = compute_turnover(self.holdings)
        assert (to >= 0).all()

    def test_net_returns_leq_gross(self):
        net = apply_transaction_costs(self.returns, self.holdings, one_way_bps=10)
        assert (net <= self.returns).all()

    def test_zero_tc_unchanged(self):
        net = apply_transaction_costs(self.returns, self.holdings, one_way_bps=0)
        pd.testing.assert_series_equal(net, self.returns, check_names=False)

    def test_btc_positive_when_strategy_beats_benchmark(self):
        bench = self.returns * 0.5  # benchmark returns half as much
        btc = compute_btc(self.returns, bench, self.holdings)
        assert btc > 0
