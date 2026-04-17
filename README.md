# StockPickingML

A production-quality Python implementation of the stock-picking strategy described in:

> **Wolff, D. & Echterling, F. (2022).**  
> *Stock Picking with Machine Learning.*  
> SSRN Working Paper 3607845.

---

## Project overview

The paper trains ML classifiers on S&P 500 constituents (1999–2021, weekly frequency) to predict whether each stock will **outperform the cross-sectional median return** over the next week.  The top-N highest-probability stocks are held equally weighted.

This repository replicates the paper's methodology as closely as possible using freely available data sources (yfinance).

---

## Project structure

```
StocKPickingProject/
├── src/
│   ├── config.py                  # Config loader (YAML → Config object)
│   ├── data/
│   │   ├── universe.py            # S&P 500 universe (current or historical CSV)
│   │   ├── prices.py              # Weekly OHLCV download via yfinance
│   │   ├── fundamentals.py        # Fundamental data (yfinance.info)
│   │   ├── loaders.py             # Assembles the weekly panel
│   │   └── caching.py             # Parquet-based cache layer
│   ├── features/
│   │   ├── technical.py           # All 21 technical indicators from the paper
│   │   ├── fundamentals.py        # Fundamental feature list + availability report
│   │   ├── preprocessing.py       # Winsorise + standardise (train-set stats only)
│   │   └── target.py              # Binary label: 1 iff return > xs median
│   ├── models/
│   │   ├── base.py                # BaseStockModel ABC + get_model() factory
│   │   ├── linear_models.py       # Ridge, Lasso, ElasticNet, PCA+LR
│   │   ├── tree_models.py         # RandomForest, XGBoost
│   │   ├── neural_models.py       # DNN (PyTorch), LSTM (PyTorch)
│   │   └── ensemble.py            # Simple average ensemble
│   ├── backtest/
│   │   ├── rolling_training.py    # Rolling 3yr train / 1yr test loop
│   │   ├── portfolio.py           # Top-N equal-weight portfolio + benchmarks
│   │   ├── metrics.py             # CAGR, Sharpe, drawdown, hit rate, CAPM α …
│   │   └── transaction_costs.py   # Turnover, net returns, BTC
│   ├── reports/
│   │   ├── charts.py              # Equity curves, drawdown, rolling Sharpe …
│   │   ├── tables.py              # Performance & picks tables
│   │   └── export.py              # Orchestrates full report generation
│   └── cli/
│       └── main.py                # Click CLI commands
├── configs/
│   └── default.yaml               # All configurable parameters
├── data/
│   ├── raw/                       # Cached parquet downloads
│   └── processed/                 # Feature panels
├── outputs/                       # Charts, CSV exports, reports
├── tests/
│   ├── test_features.py
│   ├── test_target.py
│   └── test_backtest.py
├── requirements.txt
└── pyproject.toml
```

---

## How this project approximates the paper

| Paper element | Implementation |
|---|---|
| Universe | Current S&P 500 (Wikipedia scrape) – see [Known Limitations](#known-limitations) |
| Weekly data | Wednesday close via yfinance W-WED resample |
| Technical features | All 21 from Table 1 Panel B (exact match) |
| Fundamental features | 15 of 20 paper variables via yfinance.info |
| Target | Binary: 1 if return > cross-sectional median (strict GT, configurable) |
| 3-month fundamental lag | Applied to all fundamentals (Phase-1: broadcast lag, Phase-2: time-series) |
| Training window | 3 years (156 weeks), retrain annually |
| Validation | Last 20% of training window (no future leakage) |
| Preprocessing | Forward-fill → xs median imputation → winsorise → z-score (train stats only) |
| Ridge / Lasso | sklearn LogisticRegressionCV, C ∈ [1e-4, 1e4] log-scale, 5-fold TS-CV |
| ElasticNet | Same + l1_ratio ∈ [0.05 … 1.0] |
| PCA + LR | n_components chosen by TS-CV up to 90% variance elbow |
| Random Forest | Exact paper grid (trees, depth, min_leaf), 5-fold TS-CV |
| XGBoost | Replaces AdaBoost (same grid spirit) with early stopping |
| DNN | PyTorch: 3 layers (20-10-5), BatchNorm, L1 reg, RMSprop lr=0.001 |
| LSTM | PyTorch: 1 layer 30 cells, seq_len=8 weeks, RMSprop lr=0.001 |
| Ensemble | Simple average of all model probabilities |
| Portfolio | Top-50 equal-weight long-only, weekly rebalance |
| Transaction costs | Configurable bps; BTC calculated per paper |
| Benchmarks | 1/N equal-weight universe + SPY weekly returns |

---

## Where this implementation differs from the paper

1. **Data source**: Paper uses Bloomberg; we use yfinance (free, slightly different data quality).
2. **Fundamental frequency**: Paper uses quarterly Bloomberg snapshots; Phase-1 uses the current `yfinance.info` snapshot broadcast across all weeks with a 3-month lag applied as a cutoff, not a rolling quarterly series.
3. **Universe**: Paper uses *historical* S&P 500 constituents (1,164 unique stocks since 1999); Phase-1 uses the *current* ~503 members (survivorship-biased — see [Known Limitations](#known-limitations)).
4. **Price type**: Paper uses open prices; we use adjusted close (configurable).
5. **XGBoost vs AdaBoost**: We use XGBoost with early stopping instead of AdaBoost 1000 iterations (faster, comparable results).
6. **LSTM sequence**: Paper feeds the full feature vector at each weekly step; we use a sliding window of `seq_len=8` weeks.

---

## Known limitations

### ⚠ Survivorship bias (Phase-1)

> This is the most significant caveat.  The paper explicitly uses **historical constituents** to avoid survivorship bias.  Phase-1 of this project uses the **current** S&P 500 list.  This means:
> - Only companies that survived to today are included.
> - Stocks that were added and then removed (e.g. bankruptcies, mergers) are absent.
> - Backtest results will be **overstated** relative to the paper.

**To remove survivorship bias**: provide a `historical_constituents_csv` in `configs/default.yaml`.  A suitable file has columns `[date, ticker]` with one row per (week, constituent) pair.  Such files can be sourced from the CRSP database, S&P Dow Jones Indices, or maintained community datasets.

### Other limitations

- Fundamentals from `yfinance.info` are point-in-time snapshots, not quarterly panels.
- Some paper features are unavailable in yfinance (earnings variability, employee growth, exact ROIC) — see the feature availability report after running `build-dataset`.
- The backtest does not model bid-ask spread, market impact, or slippage.

---

## Setup

### 1. Create and activate the virtual environment

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU-accelerated PyTorch (optional):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Running the pipeline

### Full pipeline (recommended first run)

```bash
python -m src.cli.main full-run
```

### Step by step

```bash
# 1. Download and cache raw data
python -m src.cli.main download-data

# 2. Build feature panel (technical + fundamentals + target)
python -m src.cli.main build-dataset

# 3. Train rolling models and run backtest
python -m src.cli.main train-backtest

# 4. Show and save current picks
python -m src.cli.main current-picks --top-n 50
```

### Options

```bash
# Use only fast models for a quick test
python -m src.cli.main train-backtest --models ridge,lasso,ensemble

# Override portfolio size
python -m src.cli.main train-backtest --top-n 100

# Use a custom config
python -m src.cli.main --config my_config.yaml full-run

# Disable transaction costs
python -m src.cli.main train-backtest --no-tc

# Force re-download all data
python -m src.cli.main download-data --force-refresh
```

---

## Configuration

All parameters live in `configs/default.yaml`.  Key sections:

```yaml
universe:
  historical_constituents_csv: null   # ← provide this to eliminate survivorship bias

data:
  start_date: "2010-01-01"            # ← extend back to 2000 for longer history
  fundamentals_lag_months: 3

rolling:
  train_years: 3
  test_years: 1

models:
  enabled: [ridge, lasso, elasticnet, pca_logistic, random_forest, xgboost, dnn, lstm, ensemble]

portfolio:
  top_n: 50

transaction_costs:
  one_way_bps: 5
```

---

## Outputs

After running `train-backtest` or `full-run`, the `outputs/` directory contains:

| File | Description |
|---|---|
| `equity_curves.png` | Cumulative returns for all models vs benchmarks |
| `drawdown.png` | Peak-to-trough drawdown chart |
| `rolling_sharpe.png` | Rolling 52-week Sharpe ratio |
| `model_comparison_table.png` | Visual metrics table |
| `metrics.csv` | Full performance metrics |
| `metrics_formatted.csv` | Human-readable formatted version |
| `predictions.parquet` | All out-of-sample predictions |
| `holdings_*.parquet` | Weekly holdings history per model |
| `current_picks.md` | Markdown report of top-50 stocks |
| `current_picks.csv` | CSV version of current picks |
| `feature_availability.csv` | Which paper features are implemented |

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover feature engineering, target construction, and backtest logic using synthetic data (no network access required).

---

## Extending the project

### Phase 2: Historical constituents

1. Obtain a historical constituent file (date, ticker CSV).
2. Set `universe.historical_constituents_csv` in your config.
3. Re-run `download-data` and `build-dataset`.

### Phase 2: Time-series fundamentals

Replace `src/data/fundamentals.py` with a quarterly panel source
(e.g. SimFin, SEC EDGAR XBRL API, or Financial Modeling Prep).  The
`align_fundamentals_to_panel` function already handles arbitrary time-series
fundamental panels.

### Phase 3: SHAP explainability

Install `shap` (already in requirements) and call:
```python
import shap
explainer = shap.TreeExplainer(rf_model._clf)
shap_values = explainer.shap_values(X_test)
```

---

## Citation

If you use this project for research, please cite the original paper:

```bibtex
@article{wolff2022stockpicking,
  title={Stock Picking with Machine Learning},
  author={Wolff, Dominik and Echterling, Fabian},
  journal={SSRN Working Paper},
  number={3607845},
  year={2022}
}
```
