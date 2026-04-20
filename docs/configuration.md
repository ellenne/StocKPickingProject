# Configuration Reference

The project is controlled by a YAML config file. `configs/default.yaml` holds
all defaults. `configs/fast.yaml` overrides only the keys that differ (deep merge).

Load order: `default.yaml` is always read first, then the user-supplied file is
merged on top using a recursive deep-merge (lists are replaced, not appended).

```bash
python -m src.cli.main --config configs/fast.yaml <command>
```

---

## `universe`

| Key | Type | Default | Description |
|---|---|---|---|
| `source` | str | `"sp500_wikipedia"` | Ticker source. Only `"sp500_wikipedia"` implemented. |
| `historical_constituents_csv` | str\|null | `null` | Path to a CSV with `[date, ticker]` for point-in-time membership. If null, current S&P 500 is used (survivorship-biased). |
| `survivorship_bias_warning` | bool | `true` | Emit a warning when no historical CSV is provided. |

---

## `data`

| Key | Type | Default | Description |
|---|---|---|---|
| `start_date` | str | `"2010-01-01"` | History start date. `fast.yaml` uses `"2015-01-01"`. |
| `end_date` | str\|null | `null` | History end date. null = today. |
| `rebalance_day` | str | `"WED"` | Rebalance day. Only WED supported per paper. |
| `price_source` | str | `"yfinance"` | Price data provider. |
| `fundamentals_source` | str | `"yfinance"` | Fundamental data provider. |
| `fundamentals_lag_months` | int | `3` | Months of publication lag applied to fundamentals. |
| `cache_dir` | str | `"data/raw"` | Directory for Parquet cache files. |
| `processed_dir` | str | `"data/processed"` | Directory for assembled panel files. |
| `max_missing_pct` | float | `0.3` | Drop tickers with more than this fraction of missing weekly close prices. |

---

## `features`

| Key | Type | Default | Description |
|---|---|---|---|
| `winsorize` | bool | `true` | Enable winsorisation before z-scoring. |
| `winsorize_pct` | float | `0.01` | Fraction clipped from each tail (1%). |
| `technical` | bool | `true` | Include technical indicator features. |
| `fundamental` | bool | `true` | Include fundamental ratio features. |
| `sector_dummies` | bool | `true` | Include one-hot sector dummies. |
| `ffill_limit` | int | `52` | Maximum consecutive weeks to forward-fill within a ticker. |

---

## `target`

| Key | Type | Default | Description |
|---|---|---|---|
| `label_strict_gt` | bool | `true` | If true, label=1 iff return > xs_median (strict); if false, label=1 iff return >= xs_median. |

---

## `rolling`

| Key | Type | Default | Description |
|---|---|---|---|
| `train_years` | int | `3` | Training window in years (paper: 3 = 156 weeks). |
| `test_years` | int | `1` | Test window in years (1 = 52 weeks). |
| `val_fraction` | float | `0.20` | Fraction of training window held back as validation. |
| `min_train_weeks` | int | `104` | Minimum training weeks required to form a window. |
| `seed` | int | `42` | Random seed for reproducibility. |

---

## `models`

### `models.enabled`

List of model names to train and evaluate. Valid names:

```yaml
models:
  enabled:
    - ridge
    - lasso
    - elasticnet
    - pca_logistic
    - random_forest
    - xgboost
    - dnn
    - lstm
    - ensemble
```

`ensemble` averages the probabilities of all other enabled models. It does not
appear in the enabled list for individual training but is included in portfolio
construction if listed.

### `models.ridge`

| Key | Type | Default | Description |
|---|---|---|---|
| `C_grid` | list[float] | `[1e-4..1e4]` | Inverse regularisation strengths to try. |
| `cv_folds` | int | `5` | TimeSeriesSplit folds. |

### `models.lasso`

Same keys as `ridge`.

### `models.elasticnet`

| Key | Type | Default | Description |
|---|---|---|---|
| `C_grid` | list[float] | `[1e-4..1e4]` | Inverse regularisation strengths. |
| `l1_ratio_grid` | list[float] | `[0.05..1.0]` | L1 fraction of penalty. |
| `cv_folds` | int | `5` | TimeSeriesSplit folds. |

### `models.pca_logistic`

| Key | Type | Default | Description |
|---|---|---|---|
| `max_components` | int\|null | `null` | Max PCA components. null = elbow (90% variance). |
| `cv_folds` | int | `5` | TimeSeriesSplit folds. |

### `models.random_forest`

| Key | Type | Default | Description |
|---|---|---|---|
| `n_estimators_grid` | list[int] | `[100, 250, 500]` | Number of trees. |
| `max_depth_grid` | list[int] | `[3, 5, 7, 10]` | Max tree depth. |
| `min_samples_leaf_grid` | list[int] | `[1, 3, 5]` | Min samples per leaf. |
| `cv_folds` | int | `5` | TimeSeriesSplit folds. |
| `n_jobs` | int | `-1` | CPU cores (-1 = all). |

### `models.xgboost`

| Key | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | int | `1000` | Max boosting rounds. |
| `early_stopping_rounds` | int | `50` | Early stopping patience. |
| `max_depth_grid` | list[int] | `[3, 5, 7]` | Tree depths. |
| `min_child_weight_grid` | list[int] | `[1, 3, 5]` | Min child weight. |
| `eta_grid` | list[float] | `[0.01, 0.05, 0.1]` | Learning rates. |
| `colsample_bytree_grid` | list[float] | `[0.7, 0.9, 1.0]` | Column subsampling. |
| `gamma_grid` | list[float] | `[0, 0.01, 0.1]` | Min split gain. |
| `subsample` | float | `0.5` | Row subsampling. |
| `cv_folds` | int | `5` | TimeSeriesSplit folds. |

### `models.dnn`

| Key | Type | Default | Description |
|---|---|---|---|
| `hidden_layers` | list[int] | `[20, 10, 5]` | Neurons per hidden layer. |
| `activation` | str | `"relu"` | Activation function. |
| `epochs` | int | `100` | Max training epochs. |
| `early_stopping_patience` | int | `10` | Epochs without improvement before stopping. |
| `learning_rate` | float | `0.001` | Adam learning rate. |
| `l1_reg` | float | `0.0001` | L1 weight regularisation. |
| `batch_norm` | bool | `true` | Apply batch normalisation after each layer. |
| `dropout` | float | `0.0` | Dropout rate (0 = disabled). |
| `batch_size` | int | `512` | Mini-batch size. |
| `seed` | int | `42` | PyTorch seed. |

### `models.lstm`

| Key | Type | Default | Description |
|---|---|---|---|
| `hidden_units` | int | `30` | LSTM hidden state dimension. |
| `seq_len` | int | `8` | Input sequence length in weeks. |
| `epochs` | int | `100` | Max training epochs. |
| `early_stopping_patience` | int | `10` | Epochs without improvement before stopping. |
| `learning_rate` | float | `0.001` | Adam learning rate. |
| `dropout` | float | `0.0` | Dropout rate. |
| `batch_size` | int | `256` | Mini-batch size. |
| `seed` | int | `42` | PyTorch seed. |

---

## `portfolio`

| Key | Type | Default | Description |
|---|---|---|---|
| `top_n` | int | `50` | Default portfolio size (paper finding). |
| `top_n_alternatives` | list[int] | `[100, 200]` | Alternative sizes evaluated in reports. |
| `equal_weight` | bool | `true` | Equal-weight portfolio (true = paper method). |

---

## `transaction_costs`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Deduct transaction costs from returns. |
| `one_way_bps` | float | `5` | One-way cost in basis points. |

---

## `outputs`

| Key | Type | Default | Description |
|---|---|---|---|
| `dir` | str | `"outputs"` | Output directory for all reports. |
| `save_predictions` | bool | `true` | Save predictions.parquet. |
| `save_holdings` | bool | `true` | Save holdings_<model>.parquet. |
| `save_charts` | bool | `true` | Save PNG/PDF/SVG charts. |
| `chart_format` | str | `"png"` | Chart file format: `png`, `pdf`, or `svg`. |