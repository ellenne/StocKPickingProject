# Feature Engineering

## Overview

Features are computed from the weekly price and fundamental panels and combined
into a MultiIndex(date, ticker) DataFrame. The pipeline applies two preprocessing
stages before model training:

1. **Panel-level:** `build_feature_matrix()` in `src/features/preprocessing.py`
   - Forward-fill within ticker (temporal imputation)
   - Cross-sectional median imputation (handles remaining NaNs)

2. **Window-level:** `FeaturePreprocessor.fit(X_train).transform(X)` 
   - Winsorisation at 1st/99th percentile (fitted on training data only)
   - Z-score standardisation (fitted on training data only)

This two-stage design ensures that no future information contaminates the
training data.

---

## Feature groups

### A. Technical indicators (`src/features/technical.py`)

All technical features are computed from the weekly close price panel.
Weekly frequency = 1 row per Wednesday.

#### Momentum

| Feature | Formula | Window |
|---|---|---|
| `mom_12m` | Cumulative log-return (skip t-1) | 52 weeks |
| `mom_6m` | Cumulative log-return (skip t-1) | 26 weeks |
| `mom_1m` | Cumulative log-return (skip t-1) | 4 weeks |
| `rel_mom_12m` | `mom_12m - SPY_mom_12m` | 52 weeks |
| `rel_mom_6m` | `mom_6m - SPY_mom_6m` | 26 weeks |
| `rel_mom_1m` | `mom_1m - SPY_mom_1m` | 4 weeks |

Momentum uses the standard Jegadeesh-Titman convention of skipping the most
recent week (`.shift(1)`) to avoid short-term reversal contamination.

#### Moving averages (price ratio)

| Feature | Formula | MA window |
|---|---|---|
| `log_price_ma200` | `log(close / MA200)` | 40 weeks (~200 days) |
| `log_price_ma100` | `log(close / MA100)` | 20 weeks (~100 days) |
| `log_price_ma50` | `log(close / MA50)` | 10 weeks (~50 days) |

#### Risk

| Feature | Formula | Window |
|---|---|---|
| `beta_12m` | OLS beta vs SPY | 52 weeks |
| `vol_12m` | Ann. std of log-returns | 52 weeks |
| `vol_6m` | Ann. std of log-returns | 26 weeks |
| `vol_1m` | Ann. std of log-returns | 4 weeks |

#### Wilder RSI (approximated at weekly frequency)

| Feature | Daily RSI | Weekly window used |
|---|---|---|
| `rsi_14` | RSI(14) | 3 weeks |
| `rsi_9` | RSI(9) | 2 weeks |
| `rsi_3` | RSI(3) | 2 weeks (min viable Wilder window) |

Note: daily RSI windows are approximated at weekly frequency by dividing by 5 and
rounding up. RSI-3 previously used window=1 (Wilder alpha=1.0 = zero memory),
producing ~55% NaN; fixed to window=2. See [Paper Alignment Audit](paper_alignment_audit.md).

RSI formula (Wilder smoothing):
```
delta  = close.diff()
gain   = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
loss   = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
RS     = gain / loss
RSI    = 100 - 100 / (1 + RS)
```

#### Bollinger Bands

| Feature | Formula | Window |
|---|---|---|
| `log_price_bb_upper` | `log(close / (MA4 + 2*std4))` | 4 weeks |
| `log_price_bb_lower` | `log(close / (MA4 - 2*std4))` | 4 weeks |

#### Lagged returns and volume

| Feature | Formula |
|---|---|
| `ret_lag1` | Log-return at t-1 |
| `ret_lag2` | Log-return at t-2 |
| `usd_volume` | `log(close * volume_week)` |

---

### B. Fundamental ratios (`src/features/fundamentals.py`, `src/data/fundamentals.py`)

All fundamental features are sourced from `yfinance.Ticker.info` (current snapshot)
and aligned across all weekly dates with a **3-month publication lag**.

| Feature | Source / derivation | Paper variable |
|---|---|---|
| `market_cap` | `marketCap` (log-transformed) | Size |
| `book_to_market` | `1/priceToBook` | Value |
| `eps_growth` | `earningsGrowth` | EPS growth |
| `leverage` | `debtToEquity` | Leverage |
| `roe` | `returnOnEquity` | ROIC (proxy) |
| `forward_eps` | `forwardEps` | Analyst forward EPS |
| `net_income_mc` | `netIncomeToCommon/marketCap` | Net income yield |
| `sales_ev` | `totalRevenue/enterpriseValue` | Sales/EV |
| `cfo_mc` | `operatingCashflow/marketCap` | CFO yield |
| `fcfe_mc` | `freeCashflow/marketCap` | FCFE yield |
| `fcf_ev` | `freeCashflow/enterpriseValue` | FCF/EV |
| `dividend_yield` | `dividendYield` | Dividend yield |
| `operating_margin` | `operatingMargins` | Operating margin |
| `profit_margin` | `profitMargins` | Profit margin |
| `revenue_growth` | `revenueGrowth` | Sales growth |

`market_cap` is log-transformed before standardisation to reduce skewness.

**Known limitation (Phase-1):** These are *static* current snapshots. For a bias-free
backtest, quarterly time-series data (SEC EDGAR / SimFin / Compustat) is required.

---

### C. Sector dummies

One-hot encoded GICS sector labels (12 categories) sourced from `yfinance.info`.
Columns are named `sector_<SectorName>` (spaces replaced with underscores).

These columns are boolean and are cast to float before winsorisation to avoid
numpy type errors.

---

## NaN treatment

The paper (Section 2) specifies a two-step process applied in this exact order:

### Step 1: Forward-fill within ticker

```python
df = df.groupby(level="ticker", group_keys=False)
       .apply(lambda g: g.ffill(limit=52))
```

Each stock's missing values are filled with its own most recent observation.
The `limit=52` cap prevents filling gaps longer than one year.
This uses only past data (no lookahead).

### Step 2: Cross-sectional median imputation

```python
def _xs_median_impute(group):
    return group.fillna(group.median())

df = df.groupby(level="date", group_keys=False).apply(_xs_median_impute)
```

Any remaining NaNs are filled with the cross-sectional median across all
stocks on that date. This is computed independently per date (no lookahead).

### NaN rates (current run, 2015-2026)

| Feature | Before imputation | After imputation |
|---|---|---|
| market_cap | 2.2% | 2.2% |
| book_to_market | 2.2% | 2.2% |
| eps_growth | 13.9% | 2.2% |
| leverage | 11.4% | 2.2% |
| dividend_yield | 19.5% | 2.2% |
| mom_12m | 9.4% | 9.0% |
| vol_12m | 9.2% | 8.8% |
| rsi_3 | 55.3% (was; fixed) | 0.2% |
| **Total avg** | **6.1%** | **2.5%** |

Residual ~2.2% NaN on fundamentals corresponds to tickers for which yfinance
returns no info at all (e.g. recently added or unusual tickers).
Residual ~9% on 12-month technical features is correct: stocks without 52 weeks
of history at the panel start cannot have a valid 12-month momentum.

---

## Standardisation (`FeaturePreprocessor`)

Fitted exclusively on the **training window** of each rolling fold:

```python
prep = FeaturePreprocessor(winsorize=True, winsorize_pct=0.01)
prep.fit(X_train)       # computes percentiles, mean, std from training data
X_train_s = prep.transform(X_train)
X_test_s  = prep.transform(X_test)  # applies training-data statistics
```

Steps:
1. Clip each column to [1st percentile, 99th percentile] of the training distribution
2. Z-score: `(x - mean_train) / std_train`
3. Fill any residual NaN with 0 (i.e., impute to training mean)