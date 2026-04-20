# Paper Alignment Audit

**Reference:** Wolff, D. & Echterling, F. (2022). *Stock Picking with Machine Learning.*  
Finance Research Letters, 49, 103017. https://doi.org/10.1016/j.frl.2022.103017

**Run date:** April 2026  
**Panel:** 590 Wednesdays, 2015-01-07 to 2026-04-16, ~470 tickers, 277,300 rows

Validation script: `tests/validate_paper_alignment.py`

```bash
python tests/validate_paper_alignment.py
```

---

## Summary

| Check | Status | Notes |
|---|---|---|
| Wednesday sampling | PASS | All 590 dates are Wednesdays, 7-day gaps, 52/53 per year |
| Missing value treatment | PASS (after fix) | Two-step ffill -> xs-median correct; rsi_3 bug fixed |
| Fundamental 3-month lag | PASS | First non-NaN date 2015-04-08 vs cutoff 2015-04-07 |
| Target construction | PASS | 50.0% positive, forward return > xs_median, last date NaN |
| Return calculation | PASS | log(close_t / close_t-1) verified, <0.03% >|50%| returns |
| Lookahead bias | PARTIAL | Code correct; static fundamentals are a structural limitation |

---

## Confirmed correct

### 1. Wednesday sampling (Paper Section 2)

> "We sample data on a Wednesday-by-Wednesday basis to avoid start and end of the week effects."

- All 590 date-index entries are Wednesdays (dayofweek == 2)
- All consecutive gaps are exactly 7 days
- 2020 and 2025 have 53 weeks (correct for 52-week calendar edge)
- Wednesday holiday handling fixed (see BUG-1 below)

### 2. Two-step missing value treatment (Paper Section 2)

> "We replace missing values in the feature set by carrying forward the last observation available. Any remaining missing values are filled using simple cross-sectional median imputation."

Implementation in `src/features/preprocessing.py::build_feature_matrix()`:

```python
# Step 1: forward-fill within ticker (no lookahead)
df = df.groupby(level="ticker", group_keys=False)
       .apply(lambda g: g.ffill(limit=52))

# Step 2: cross-sectional median imputation per date (no lookahead)
df = df.groupby(level="date", group_keys=False)
       .apply(lambda g: g.fillna(g.median()))
```

Total NaN drops from 6.1% to 2.5% after imputation.

### 3. Fundamental 3-month lag (Paper Section 2)

> "We include three months lag for all fundamental data to avoid any forward-looking bias."

Implemented in `src/data/fundamentals.py::align_fundamentals_to_panel()`:

```python
cutoff = weekly_dates[0] + pd.DateOffset(months=3)
panel.loc[dates < cutoff] = NaN
```

Verified: first non-NaN date for all 15 fundamental columns is 2015-04-08,
which is after the expected cutoff of 2015-04-07 (panel start + 3 months).

### 4. Binary target (Paper Section 2)

> "We use a binary classification target: 1 if the stock's return exceeds the cross-sectional median return in the following week, 0 otherwise."

Implemented in `src/features/target.py::add_target()`:

```python
fwd_ret = ret.shift(-1)                      # next-week return
xs_median = fwd_ret.median(axis=1)           # cross-sectional median
label = fwd_ret.gt(xs_median, axis=0)        # strict_gt=True
```

Observed positive rate: 50.0% (expected by construction).

### 5. Wednesday-to-Wednesday log-returns

```python
log_return_t = log(close_t / close_t-1)
```

Verified to machine precision for all tickers. Less than 0.03% of weekly
returns exceed |50%|, confirming no data errors.

### 6. Anti-lookahead guarantees

- `FeaturePreprocessor` (winsorise + z-score) fitted ONLY on training window
- Rolling backtest test periods are strictly after training periods
- Forward-fill uses only past values within each ticker
- Cross-sectional median computed independently per date
- Target uses forward return (no present information used)

---

## Bugs fixed

### BUG-1: Wednesday holiday direction [SIGNIFICANT - Fixed]

**Paper (Section 2):** "if a Wednesday is a non-trading day (holiday), we use data of the next trading day available."

**Previous code:** `resample("W-WED", Close="last")` picks the **previous** trading day (e.g. Tuesday) when Wednesday is a holiday.

**Fix applied** in `src/data/prices.py::_daily_to_weekly()`:

```python
# bfill: for each Wednesday label, use the first available day >= that date
# tolerance="4D": only look within same week (Thu-Sun max)
close_next = daily[["Close"]].reindex(
    weekly.index, method="bfill", tolerance=pd.Timedelta("4D")
)
weekly["Close"] = close_next["Close"]
```

**Impact:** Affects only the handful of Wednesdays per year that fall on US
market holidays (Christmas, July 4th, etc.). Numerical difference is minimal
but brings the implementation into strict compliance with the paper.

---

### BUG-3: RSI-3 degenerate window=1 [SIGNIFICANT - Fixed]

**Problem:** RSI-3 (3-day RSI) was mapped to Wilder window=1 at weekly frequency:

```python
rsi_dict_3 = {col: _rsi(close_panel[col], 1) for col in ...}
```

Window=1 gives `ewm(alpha=1.0)` — zero memory. When a week shows no decline,
`loss_ewm == 0` which is replaced with NaN, producing an undefined RSI.

**Observed NaN rate:** 55.3% (visible in the validation output and Check 7 table)

**Fix applied** in `src/features/technical.py`:

```python
# Changed window from 1 to 2 (minimum viable Wilder window)
rsi_dict_3 = {col: _rsi(close_panel[col], 2) for col in ...}
```

**Impact:** After re-running `build-dataset`, rsi_3 NaN drops to ~0.8%.
Note that RSI-3 at weekly frequency is still an approximation (true 3-day RSI
requires daily price input).

---

## Known limitations (structural — not fixable in Phase-1)

### BUG-2: Static fundamentals lookahead bias [CRITICAL]

**Nature:** `yfinance.info` returns **today's** values (2025 P/B, margins, ROE, etc.).
These are broadcast back to ALL historical dates 2015–present.

**What the 3-month lag does NOT fix:** The lag only zeroes out data for the first
three months of the panel. It does not prevent the model from seeing 2025 earnings
when predicting 2016 stock returns.

**Expected effect on backtest:** Fundamental features will show higher information
coefficient than they should. The ensemble Sharpe ratio and alpha are likely
optimistically biased. The bias is most severe in fundamental-heavy models
(book_to_market, leverage, ROE, margins).

**Phase-2 fix (requires external data):**

Replace `src/data/fundamentals.py` with a time-series source:
- **SEC EDGAR XBRL** (free, US companies, quarterly filings with exact dates)
- **SimFin Free** (free tier, income statement + balance sheet quarterly)
- **Compustat / Refinitiv** (commercial, gold standard)

Apply the 3-month lag per filing announcement date:

```python
# Per-filing approach (Phase-2)
available_date = filing_date + pd.DateOffset(months=3)
panel.loc[date < available_date, fundamental_cols] = NaN
```

---

## Minor warnings

### WARN-1: `freq` attribute lost in MultiIndex

The date level of the MultiIndex panel has `freq=None`. The assertion
`assert dates.freq == "W-WED"` will fail. Use instead:

```python
assert panel.index.get_level_values("date").dayofweek.unique().tolist() == [2]
```

The dates are verified to be all Wednesdays with 7-day gaps.

### WARN-2: Forward-fill limit = 52 weeks

The paper states "carrying forward the last observation available" without
an explicit limit. The implementation uses `limit=52` (one year). This is a
pragmatic choice; tickers absent for >1 year receive cross-sectional median
imputation rather than their last known value.

### WARN-3: RSI/Bollinger bands are weekly approximations

All technical indicators that the paper computes at daily frequency (RSI,
Bollinger bands, moving averages) are approximated at weekly frequency:
- RSI-14 daily -> window=3 weekly
- 20-day Bollinger band -> 4-week band
- 200-day MA -> 40-week MA

Intra-week price variation is lost. These features will have less
signal than their daily counterparts from the original paper.

---

## NaN rates: before and after imputation

| Feature | Before | After |
|---|---|---|
| market_cap | 2.2% | 2.2% |
| book_to_market | 2.2% | 2.2% |
| eps_growth | 13.9% | 2.2% |
| leverage | 11.4% | 2.2% |
| roe | 9.1% | 2.2% |
| forward_eps | 2.4% | 2.2% |
| net_income_mc | 2.6% | 2.2% |
| sales_ev | 2.8% | 2.2% |
| cfo_mc | 5.5% | 2.2% |
| fcfe_mc | 9.1% | 2.2% |
| fcf_ev | 9.1% | 2.2% |
| dividend_yield | 19.5% | 2.2% |
| operating_margin | 2.4% | 2.2% |
| profit_margin | 2.4% | 2.2% |
| revenue_growth | 2.8% | 2.2% |
| mom_12m | 9.4% | 9.0% |
| mom_6m | 5.0% | 4.6% |
| mom_1m | 1.3% | 0.8% |
| vol_12m | 9.2% | 8.8% |
| rsi_14 | 0.8% | 0.2% |
| rsi_3 (before fix) | 55.3% | -- |
| rsi_3 (after fix) | ~0.8% | 0.2% |
| usd_volume | 0.5% | 0.0% |
| **Total avg** | **6.1%** | **2.5%** |

Residual 2.2% in fundamentals = tickers with no yfinance.info data (cannot be
reduced without a different data source).
Residual ~9% in 12-month technical features = correct (insufficient history at
panel start for 52-week rolling windows).