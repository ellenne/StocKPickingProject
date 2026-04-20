# Backtesting

## Rolling window procedure

Following the paper (Figure 1), the backtest uses a **rolling 3-year / 1-year**
training / test split.

```
Timeline:
 2015  2016  2017  2018  2019  2020  2021  2022  2023  2024  2025
 |------ TRAIN (156 wks) ------|---- TEST (52 wks) ----|
                 |------ TRAIN (156 wks) ------|---- TEST (52 wks) ----|
                                 |-- ...
```

- **Training window:** 3 years = 156 weeks
- **Test window:** 1 year = 52 weeks
- **Slide:** 1 year (the window slides forward by one test period)
- **Validation split:** last 20% of each training window (used for early stopping
  in DNN / LSTM)
- **Minimum training rows:** 100 labelled samples (safety check)

With `fast.yaml` data starting 2015 and running to April 2026, there are
**7 rolling windows** covering test periods from 2018 to 2025.

Implementation: `src/backtest/rolling_training.py::run_rolling_backtest()`

---

## Anti-lookahead guarantees

At each window, the pipeline strictly enforces:

1. `FeaturePreprocessor.fit()` called only on training rows
2. Hyperparameter CV (`GridSearchCV` / `LogisticRegressionCV`) uses only training rows
3. Feature forward-fill and cross-sectional median are applied to the whole panel
   before splitting, but both are causally correct:
   - Forward-fill: each cell uses only past values
   - Cross-sectional median: each date uses only same-date cross-section
4. Test forward returns are never visible to the model during training

---

## Portfolio construction

### Top-N selection

At each test date `t`, the model assigns probability `P(outperform)` to each ticker.
The top-N = 50 tickers by probability are selected.

```python
# Implemented in src/backtest/portfolio.py
top_n_tickers = prob_series.nlargest(n).index
```

### Equal weighting

Each selected stock receives weight `1/N`. No position sizing, no leverage.

### Rebalancing

The portfolio is rebalanced **every week** (each Wednesday). The portfolio return
for week `t` is the equal-weighted average of the forward log-returns of the
selected stocks:

```python
port_return_t = holdings_t.dot(fwd_returns_t)   # inner product with equal weights
```

### Benchmark

Two benchmarks are computed:
- **Benchmark 1/N:** Equal-weight portfolio of ALL stocks in the universe
- **SPY:** S&P 500 ETF weekly log-returns (downloaded separately from yfinance)

---

## Performance metrics (`src/backtest/metrics.py`)

All metrics are computed **out-of-sample** on test periods only.

| Metric | Symbol | Description |
|---|---|---|
| Annualised return (gross) | CAGR | `exp(sum(r) * 52 / n) - 1` |
| Annualised return (net) | CAGR_net | After transaction costs |
| Annualised volatility | Vol | `std(r) * sqrt(52)` |
| Sharpe ratio (gross) | SR | `mean(r) * 52 / (std(r) * sqrt(52))`, rf=0 |
| Sharpe ratio (net) | SR_net | After transaction costs |
| Maximum drawdown | MDD | Peak-to-trough; `min((cum - peak) / peak)` |
| CAPM alpha | Alpha | Intercept of OLS(strat ~ market) * 52 |
| CAPM beta | Beta | Slope of OLS(strat ~ SPY) |
| Annualised turnover | TO | `mean(weekly_turnover) * 52` (one-sided) |
| Hit rate | HR | Fraction of correct outperform/underperform predictions |
| Top-10% accuracy | Acc10 | Avg accuracy in top + bottom 10% by probability |
| Top-5% accuracy | Acc5 | Avg accuracy in top + bottom 5% |
| Top-1% accuracy | Acc1 | Avg accuracy in top + bottom 1% |
| Break-to-cost BTC | BTC | Basis points per one-way trade at which strategy breaks even |

Risk-free rate is set to 0 per the paper.

---

## Transaction costs (`src/backtest/transaction_costs.py`)

| Parameter | Default | Config key |
|---|---|---|
| Enabled | true | `transaction_costs.enabled` |
| One-way cost | 5 bps | `transaction_costs.one_way_bps` |

One-way cost = 5 basis points per trade applies to both buys and sells.

Weekly turnover is computed as the sum of absolute weight changes:

```
turnover_t = sum(|w_t - w_{t-1}|) / 2
```

Net return:
```
net_return_t = gross_return_t - turnover_t * one_way_bps / 10000
```

**Break-to-cost (BTC):** the one-way transaction cost in basis points at which
the strategy's net return equals the benchmark. A higher BTC indicates a more
robust strategy.

---

## Output reports

All outputs are written to the `outputs/` directory (configured via `outputs.dir`).

| File | Description |
|---|---|
| `predictions.parquet` | (date, ticker) x (prob columns, target, fwd_return) |
| `holdings_<model>.parquet` | (date) x (ticker) equal-weight holdings |
| `performance_table.csv` | All metrics for all models |
| `equity_curve.png` | Cumulative return vs SPY and 1/N benchmark |
| `drawdown.png` | Drawdown series for all models |
| `rolling_sharpe.png` | Rolling 12-month Sharpe ratio |
| `current_picks.csv` | Latest top-N with model probabilities |
| `current_picks.md` | Human-readable picks table in Markdown |
| `feature_availability.csv` | NaN report for all feature columns |