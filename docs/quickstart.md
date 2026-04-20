# Quick Start

## Prerequisites

- Python 3.10 or later
- Git
- ~4 GB free disk space (price data + model artifacts)
- Internet connection for first data download

---

## 1. Clone and create environment

```bash
git clone https://github.com/ellenne/StocKPickingProject.git
cd StocKPickingProject

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Run the full pipeline (fast config)

```bash
python -m src.cli.main --config configs/fast.yaml full-run
```

This single command executes all four stages in sequence:

1. **download-data** – scrapes S&P 500 tickers from Wikipedia, then downloads ~10 years of weekly OHLCV prices and fundamental snapshots via yfinance.
2. **build-dataset** – assembles the weekly panel, computes technical features, applies the 3-month fundamental lag, and labels each observation.
3. **train-backtest** – runs the rolling 3-yr/1-yr training loop for all enabled models, builds top-50 portfolios, and computes performance metrics.
4. **current-picks** – outputs the latest ensemble top-50 picks.

Expected runtime with `fast.yaml`: 45–90 minutes (bottleneck: yfinance fundamental downloads ~470 tickers).

---

## 3. Run stages individually

```bash
# Download raw data only
python -m src.cli.main --config configs/fast.yaml download-data

# Assemble feature panel (requires download-data first)
python -m src.cli.main --config configs/fast.yaml build-dataset

# Train models and generate backtest report (requires build-dataset first)
python -m src.cli.main --config configs/fast.yaml train-backtest

# Show current top-50 picks (requires train-backtest first)
python -m src.cli.main --config configs/fast.yaml current-picks --top-n 50
```

---

## 4. Key output files

| File | Description |
|---|---|
| `outputs/current_picks.csv` | Top-N ticker list with model probabilities |
| `outputs/current_picks.md` | Human-readable picks in Markdown |
| `outputs/equity_curve.png` | Portfolio vs SPY equity curves |
| `outputs/drawdown.png` | Drawdown comparison chart |
| `outputs/rolling_sharpe.png` | 12-month rolling Sharpe ratios |
| `outputs/performance_table.csv` | All performance metrics for every model |
| `outputs/predictions.parquet` | Full out-of-sample probability predictions |
| `outputs/holdings_<model>.parquet` | Weekly holdings for each model |

---

## 5. Validate methodology against the paper

```bash
python tests/validate_paper_alignment.py
```

See [Paper Alignment Audit](paper_alignment_audit.md) for a full report of findings.

---

## 6. Use a custom date range or subset of models

```bash
# Only Ridge + XGBoost + Ensemble, top-100 portfolio
python -m src.cli.main --config configs/fast.yaml \
    train-backtest --models ridge,xgboost,ensemble --top-n 100

# Verbose logging
python -m src.cli.main -v --config configs/fast.yaml full-run

# Force re-download (ignore cache)
python -m src.cli.main --config configs/fast.yaml download-data --force-refresh
```

---

## 7. Run the test suite

```bash
pytest tests/
```

Individual test files:

| File | Coverage |
|---|---|
| `tests/test_features.py` | Feature matrix construction and imputation |
| `tests/test_target.py` | Binary target labelling |
| `tests/test_backtest.py` | Metrics, transaction costs, portfolio construction |
| `tests/validate_paper_alignment.py` | Paper alignment checks (run standalone) |

---

## Windows PowerShell note

If you see `UnicodeEncodeError` when running commands, set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```