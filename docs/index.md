# StocKPickingProject – Documentation

> ML-based stock picker inspired by **Wolff & Echterling (2022)**  
> "Stock Picking with Machine Learning"

---

## Contents

| Document | Description |
|---|---|
| [Quick Start](quickstart.md) | Installation, environment setup, and first full run |
| [Architecture](architecture.md) | Directory layout, module map, and end-to-end data flow |
| [Data Pipeline](data_pipeline.md) | Universe, price download, fundamentals, caching, `input/` store |
| [Feature Engineering](features.md) | Technical indicators, fundamental ratios, sector dummies, NaN treatment |
| [Models](models.md) | Ridge, Lasso, ElasticNet, PCA-Logistic, Random Forest, XGBoost, DNN, LSTM, Ensemble |
| [Backtesting](backtest.md) | Rolling-window procedure, portfolio construction, performance metrics |
| [Configuration Reference](configuration.md) | Every YAML key in `default.yaml` and `fast.yaml` explained |
| [CLI Reference](cli_reference.md) | All command-line commands with options and examples |
| [Paper Alignment Audit](paper_alignment_audit.md) | Validation of implementation vs the paper; known deviations and fixes |
| [Data Quality Report](data_quality_report.md) | Step-by-step NaN transition table, correct vs wrong xs-median, coverage timeline |

---

## Project at a glance

```
Universe → Download → Feature Engineering → Rolling ML → Portfolio → Report
  S&P 500    yfinance   Technicals             3-yr train   Top-50 EW   Equity curve
             prices     Fundamentals           1-yr test    weekly      Current picks
             fundams    Sector dummies                      rebalance   Markdown / PNG
```

### Key numbers (fast.yaml run, 2015-2025)

| Item | Value |
|---|---|
| Universe | ~470 current S&P 500 tickers |
| Date range | 2015-01-07 to 2026-04-16 |
| Weekly observations | 590 Wednesdays |
| Panel rows (date × ticker) | ~277,000 |
| Features | 36 (21 technical + 15 fundamental) + 12 sector dummies |
| Target positive rate | 50.0 % (by construction) |
| Rolling windows | 7 (3-yr train / 1-yr test, starting 2018) |

---

## Reference

**Paper:** Wolff, D. & Echterling, F. (2022).  
*Stock Picking with Machine Learning.*  
Finance Research Letters, 49, 103017.  
<https://doi.org/10.1016/j.frl.2022.103017>