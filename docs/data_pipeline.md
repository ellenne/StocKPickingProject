# Data Pipeline

## Overview

```
Wikipedia -> Universe (tickers)
yfinance  -> Prices (OHLCV, weekly W-WED)
yfinance  -> Fundamentals (snapshot)
              |
           Parquet cache (data/raw/)
           CSV cache     (input/)      <- optional, shareable
              |
           Weekly panel (data/processed/weekly_panel.parquet)
```

---

## 1. Universe (`src/data/universe.py`)

**Source:** Current S&P 500 from Wikipedia  
`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`

The scraper uses `requests` with a Chrome User-Agent header (plain `urllib` gets
HTTP 403). The HTML is passed to `pd.read_html()`.

**Survivorship bias warning:** The default configuration uses the *current*
S&P 500 list for all historical dates. This means the backtest is biased toward
companies that survived to today. The config key
`universe.historical_constituents_csv` accepts a CSV with columns
`[date, ticker]` representing point-in-time membership; if provided, survivorship
bias is eliminated.

```yaml
universe:
  source: "sp500_wikipedia"
  historical_constituents_csv: null   # set to path for point-in-time data
  survivorship_bias_warning: true
```

---

## 2. Price Download (`src/data/prices.py`)

**Source:** yfinance (`auto_adjust=True`, splits and dividends adjusted)  
**Frequency:** Daily download, resampled to **weekly Wednesday** frequency

### Wednesday resampling

Per the paper (Section 2): data is sampled on Wednesdays to avoid start/end-of-week
effects. If Wednesday is a non-trading day (holiday), the **next** available trading
day is used.

**Implementation:**

```python
# OHLV: period-based aggregates (Mon-to-Wed window)
agg = {"Open": "first", "High": "max", "Low": "min", "Volume": "sum"}
weekly = daily.resample("W-WED").agg(agg)

# Close: paper says "next trading day available" if Wednesday is a holiday
# bfill maps each Wednesday label to the first trading day >= that date
# tolerance="4D" prevents crossing into the following Wednesday
close_next = daily[["Close"]].reindex(
    weekly.index, method="bfill", tolerance=pd.Timedelta("4D")
)
weekly["Close"] = close_next["Close"]
```

Note: earlier versions used `resample(..., Close="last")` which incorrectly picked
the *previous* trading day on holidays. This was fixed in BUG-1 of the audit.

### Batching

Tickers are downloaded in batches of 100 (configurable) to avoid yfinance timeouts.
Results are cached as Parquet per batch in `data/raw/`.

### Missing tickers

Tickers that return empty DataFrames (e.g. delisted, name changes) are silently
skipped with a warning. Tickers with more than `data.max_missing_pct = 30%` missing
weekly observations are dropped before panel assembly.

---

## 3. Fundamentals (`src/data/fundamentals.py`)

**Source:** `yfinance.Ticker(ticker).info` - a JSON snapshot with ~150 fields.

### Fundamental fields extracted

| Our column | yfinance key | Notes |
|---|---|---|
| `market_cap` | `marketCap` | USD |
| `book_to_market` | `1/priceToBook` | Derived |
| `eps_growth` | `earningsGrowth` | YoY |
| `leverage` | `debtToEquity` | D/E ratio |
| `roe` | `returnOnEquity` | Proxy for ROIC |
| `forward_eps` | `forwardEps` | Analyst consensus |
| `net_income_mc` | `netIncomeToCommon/marketCap` | Derived |
| `sales_ev` | `totalRevenue/enterpriseValue` | Derived |
| `cfo_mc` | `operatingCashflow/marketCap` | Derived |
| `fcfe_mc` | `freeCashflow/marketCap` | Derived |
| `fcf_ev` | `freeCashflow/enterpriseValue` | Derived |
| `dividend_yield` | `dividendYield` | % |
| `operating_margin` | `operatingMargins` | % |
| `profit_margin` | `profitMargins` | % |
| `revenue_growth` | `revenueGrowth` | YoY |

### 3-month publication lag

Fundamental data is lagged 3 months to avoid forward-looking bias. In Phase-1, this
is implemented by setting the first `lag_months` months of the panel to NaN:

```python
cutoff = weekly_dates[0] + pd.DateOffset(months=3)
panel.loc[dates < cutoff] = NaN
```

**Phase-1 limitation:** `yfinance.info` returns *current* (today's) values, not
historical quarterly data. This means the model uses 2025 margins when predicting
2017 returns – a structural lookahead bias. The 3-month cutoff only eliminates the
very first quarter of data, not the broader bias.

**Phase-2 fix:** Replace with quarterly time-series from SEC EDGAR XBRL, SimFin,
or Compustat, applying the 3-month lag per actual filing date.

---

## 4. Caching (`src/data/caching.py`)

All network downloads are cached as Parquet files in `data/raw/`. The cache key
is a string derived from the function name and parameters. On re-run, if the
cache file exists and `force_refresh=False`, the file is loaded instead of
re-downloading.

```python
# Usage pattern
data = cached_download(
    key="prices_batch_0_2015-01-01_2026-04-16",
    fetch_fn=lambda: yf.download(...),
    cache_dir="data/raw",
    force_refresh=False,
)
```

---

## 5. CSV Input Store (`src/data/input_store.py`)

The `InputDataStore` class provides an optional human-readable CSV cache in
`input/`. It is designed for three use cases:

1. **Inspection** – open `input/prices_close.csv` in Excel to verify raw data
2. **Sharing** – commit-free data sharing without re-downloading
3. **Incremental updates** – `prices_last_date()` returns the most recent date
   already stored; `download-data` then fetches only the missing tail

### CSV layout

```
input/
  universe.csv          one column: ticker
  prices_close.csv      wide panel: rows = Wednesday dates, cols = tickers
  prices_open.csv
  prices_high.csv
  prices_low.csv
  prices_volume.csv
  fundamentals.csv      tickers x fundamental fields (latest snapshot)
  sectors.csv           two columns: ticker, sector
```

### Incremental price update logic

```python
store = InputDataStore("input/")
last = store.prices_last_date()   # e.g. 2026-04-09
# download only 2026-04-09 to today
new_prices = download_prices(tickers, start=last, end=today)
store.save_prices(new_prices_panel, "close")  # merges with existing, deduplicates
```

The `input/` folder is in `.gitignore` (raw data should not be committed).
A placeholder `input/.gitkeep` is tracked to create the directory on clone.

---

## 6. Panel Assembly (`src/data/loaders.py`)

`build_dataset(cfg)` orchestrates:

1. Load universe tickers
2. Download prices -> `build_price_panel()` -> wide close panel (date x ticker)
3. Drop tickers with >30% missing weeks
4. Compute weekly log-returns: `log(close_t / close_t-1)`
5. Download fundamentals for all tickers, reindex to surviving tickers
6. `align_fundamentals_to_panel()` -> broadcasts static snapshot across all dates
   with 3-month lag
7. Download sector classifications
8. Stack into MultiIndex(date, ticker) panel
9. Save as `data/processed/weekly_panel.parquet`