# CLI Reference

Entry point: `python -m src.cli.main`

## Global options

```
python -m src.cli.main [OPTIONS] COMMAND [ARGS]

Options:
  --config PATH    YAML config file  [default: configs/default.yaml]
  -v, --verbose    Enable DEBUG logging
  --help           Show help and exit
```

---

## Commands

### `download-data`

Download and cache raw price, fundamental, and sector data.

```bash
python -m src.cli.main --config configs/fast.yaml download-data [OPTIONS]

Options:
  --force-refresh    Re-download even if cache files already exist
```

**What it does:**
1. Scrapes current S&P 500 tickers from Wikipedia (~470 tickers)
2. Downloads daily OHLCV via yfinance (batches of 100) and caches as Parquet
3. Downloads `yfinance.info` fundamentals for each ticker and caches

**Output:** Files in `data/raw/` (Parquet format, gitignored)

**Example:**
```bash
# First run
python -m src.cli.main --config configs/fast.yaml download-data

# Re-download everything
python -m src.cli.main --config configs/fast.yaml download-data --force-refresh
```

---

### `build-dataset`

Assemble the processed weekly panel from cached raw data.

```bash
python -m src.cli.main --config configs/fast.yaml build-dataset [OPTIONS]

Options:
  --force-refresh    Rebuild from raw download even if processed files exist
```

**Requires:** `download-data` completed (cache files in `data/raw/`)

**What it does:**
1. Loads and joins price + fundamental + sector raw data
2. Drops tickers with >30% missing weeks
3. Computes 21 technical features (momentum, RSI, vol, etc.)
4. Aligns fundamentals with 3-month lag
5. Labels each observation with the binary target
6. Saves `data/processed/features_panel.parquet`
7. Writes `outputs/feature_availability.csv`

**Output:** `data/processed/features_panel.parquet` (MultiIndex date x ticker, ~277k rows)

**Example:**
```bash
python -m src.cli.main --config configs/fast.yaml build-dataset
```

---

### `train-backtest`

Run the rolling ML training loop and generate performance reports.

```bash
python -m src.cli.main --config configs/fast.yaml train-backtest [OPTIONS]

Options:
  --models TEXT     Comma-separated model names  [default: all enabled in config]
  --top-n INTEGER   Portfolio size override  [default: config value]
  --no-tc           Disable transaction cost deduction
```

**Requires:** `build-dataset` completed (`features_panel.parquet` exists)

**What it does:**
1. Loads `features_panel.parquet`
2. Applies `build_feature_matrix()` (ffill + xs-median imputation)
3. For each rolling window:
   - Fits `FeaturePreprocessor` on training data
   - Trains all enabled models
   - Generates out-of-sample predictions
4. Builds top-N portfolios for each model
5. Computes all performance metrics
6. Generates charts, tables, and the current picks report

**Output files in `outputs/`:**
- `predictions.parquet`
- `holdings_<model>.parquet`
- `performance_table.csv`
- `equity_curve.png`, `drawdown.png`, `rolling_sharpe.png`
- `current_picks.csv`, `current_picks.md`

**Examples:**
```bash
# All enabled models, top-50
python -m src.cli.main --config configs/fast.yaml train-backtest

# Ridge + Ensemble only, top-100
python -m src.cli.main --config configs/fast.yaml \
    train-backtest --models ridge,ensemble --top-n 100

# Without transaction cost deduction
python -m src.cli.main --config configs/fast.yaml train-backtest --no-tc
```

---

### `current-picks`

Display and save the latest top-N stock picks from the most recent predictions.

```bash
python -m src.cli.main --config configs/fast.yaml current-picks [OPTIONS]

Options:
  --top-n INTEGER   Number of stocks to show  [default: 50]
  --model TEXT      Model to rank by  [default: ensemble]
```

**Requires:** `train-backtest` completed (`outputs/predictions.parquet` exists)

**What it does:**
1. Loads the most recent week's predictions from `predictions.parquet`
2. Ranks tickers by the specified model's probability
3. Prints a Rich table in the terminal
4. Saves `outputs/current_picks.csv` and `outputs/current_picks.md`

**Examples:**
```bash
# Top-50 by ensemble
python -m src.cli.main --config configs/fast.yaml current-picks

# Top-20 by XGBoost
python -m src.cli.main --config configs/fast.yaml \
    current-picks --top-n 20 --model xgboost
```

---

### `full-run`

Execute the complete pipeline in sequence.

```bash
python -m src.cli.main --config configs/fast.yaml full-run [OPTIONS]

Options:
  --force-refresh    Re-download raw data
  --models TEXT      Comma-separated model names
  --top-n INTEGER    Portfolio size
```

**Equivalent to:**
```bash
download-data -> build-dataset -> train-backtest -> current-picks
```

**Example:**
```bash
python -m src.cli.main --config configs/fast.yaml full-run
```

---

## Validate paper alignment

```bash
python tests/validate_paper_alignment.py
python tests/validate_paper_alignment.py --panel data/processed/features_panel.parquet
```

Runs 7 automated checks against Wolff & Echterling (2022) methodology.
See [Paper Alignment Audit](paper_alignment_audit.md).