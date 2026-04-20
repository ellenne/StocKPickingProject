# Architecture

## Directory layout

```
StocKPickingProject/
|
|-- configs/
|   |-- default.yaml          Full-history config (2010-present, all models)
|   +-- fast.yaml             Quick-run override (2015-present, reduced grids)
|
|-- data/
|   |-- raw/                  Parquet cache for yfinance downloads (gitignored)
|   +-- processed/            Assembled weekly panel parquet files (gitignored)
|
|-- docs/                     Project documentation (this folder)
|
|-- input/                    CSV input store for reproducible data sharing (gitignored)
|
|-- outputs/                  Reports, charts, predictions (gitignored except .gitkeep)
|
|-- src/
|   |-- config.py             YAML loader and Config dataclass
|   |-- cli/
|   |   +-- main.py           Click CLI entry point (5 commands)
|   |-- data/
|   |   |-- universe.py       S&P 500 ticker list (Wikipedia scrape)
|   |   |-- prices.py         Weekly OHLCV download and resampling
|   |   |-- fundamentals.py   yfinance.info snapshot and alignment
|   |   |-- loaders.py        Orchestrator: builds the raw weekly panel
|   |   |-- caching.py        Parquet-based cache helpers
|   |   +-- input_store.py    CSV-based input/ folder manager
|   |-- features/
|   |   |-- technical.py      21 technical indicators (momentum, RSI, BB, vol, ...)
|   |   |-- fundamentals.py   15 fundamental ratio columns and feature names
|   |   |-- preprocessing.py  ffill + xs-median + winsorise + z-score
|   |   +-- target.py         Binary label: fwd_return > xs_median
|   |-- models/
|   |   |-- base.py           BaseStockModel ABC
|   |   |-- linear_models.py  Ridge, Lasso, ElasticNet, PCA-Logistic
|   |   |-- tree_models.py    RandomForest, XGBoost
|   |   |-- neural_models.py  DNN, LSTM (PyTorch)
|   |   +-- ensemble.py       Simple probability-average ensemble
|   |-- backtest/
|   |   |-- rolling_training.py  Rolling 3yr-train / 1yr-test loop
|   |   |-- portfolio.py         Top-N selection, SPY benchmark
|   |   |-- metrics.py           CAGR, Sharpe, MDD, CAPM alpha, hit rate, ...
|   |   +-- transaction_costs.py Turnover computation, BTC metric
|   +-- reports/
|       |-- charts.py         Equity curve, drawdown, rolling Sharpe plots
|       |-- tables.py         Performance table, feature availability, current picks
|       +-- export.py         Full report orchestrator
|
+-- tests/
    |-- test_features.py
    |-- test_target.py
    |-- test_backtest.py
    +-- validate_paper_alignment.py
```

---

## End-to-end data flow

```
(1) UNIVERSE
    universe.py                Wikipedia scrape -> list[ticker]
         |
(2) DOWNLOAD
    prices.py                  yfinance OHLCV daily -> weekly resample (W-WED)
    fundamentals.py            yfinance.info snapshot
         |
    caching.py                 Parquet cache (data/raw/)
    input_store.py             CSV cache (input/)  [optional]
         |
(3) ASSEMBLE PANEL
    loaders.py                 join prices + fundamentals + sectors
                               -> MultiIndex(date, ticker) parquet
                               -> data/processed/weekly_panel.parquet
         |
(4) FEATURE ENGINEERING
    technical.py               21 technical indicators
    features/fundamentals.py   15 fundamental ratio columns
    preprocessing.py           ffill -> xs-median imputation
    target.py                  binary label (fwd_return > xs_median)
                               -> data/processed/features_panel.parquet
         |
(5) ROLLING BACKTEST
    rolling_training.py        For each window:
                                 FeaturePreprocessor.fit(X_train)
                                 model.fit(X_train, y_train)
                                 proba = model.predict_proba(X_test)
         |
(6) PORTFOLIO & METRICS
    portfolio.py               sort by proba, take top-N, equal weight
    metrics.py                 CAGR, Sharpe, MDD, CAPM alpha, hit rate
    transaction_costs.py       turnover, BTC, net returns
         |
(7) REPORTS
    charts.py                  PNG: equity curve, drawdown, rolling Sharpe
    tables.py                  CSV/MD: performance table, current picks
    export.py                  Orchestrates all output files
```

---

## Key design principles

**1. No lookahead bias**  
The `FeaturePreprocessor` (winsorise + z-score) is fitted exclusively on the
training window. Forward-fill uses only past values. The target is the *forward*
(next-week) return. Rolling windows are strictly non-overlapping on the test side.

**2. Config-driven**  
Every hyperparameter, date range, feature toggle, model grid, and output path is
controlled by a single YAML file. Production runs use `default.yaml`; development
uses `fast.yaml`.

**3. Pluggable data layer**  
The `prices.py` / `fundamentals.py` modules expose a simple download interface.
Replacing yfinance with a commercial provider (e.g. FMP, Refinitiv) requires only
modifying those two files.

**4. Parquet-first, CSV-optional**  
Internal data is cached as compressed Parquet for performance. The `InputDataStore`
class in `input_store.py` additionally exports everything as plain CSVs to `input/`
for inspection in Excel and incremental updates.

**5. Paper faithful where possible**  
Implementation follows Wolff & Echterling (2022) Section 2 methodology:
Wednesday sampling, two-step NaN imputation, 3-month fundamental lag,
binary cross-sectional target, rolling 3yr/1yr windows.
See [Paper Alignment Audit](paper_alignment_audit.md) for known deviations.