# Models

## Overview

All models implement `BaseStockModel` (`src/models/base.py`) with a common interface:

```python
model.fit(X_train, y_train, X_val=None, y_val=None)
proba = model.predict_proba(X_test)   # shape (n_samples, 2); column 1 = P(outperform)
```

Models receive **standardised** features (output of `FeaturePreprocessor.transform()`).
Hyperparameter selection uses `TimeSeriesSplit` CV on the training window.

---

## Linear models (`src/models/linear_models.py`)

### Ridge Logistic Regression

Logistic regression with L2 regularisation.

```yaml
models:
  ridge:
    C_grid: [0.001, 0.01, 0.1, 1, 10]   # fast.yaml (default: 1e-4 to 1e4)
    cv_folds: 3                           # fast.yaml (default: 5)
```

- Solver: `liblinear` (fast for small datasets)
- Hyperparameter: `C` (inverse regularisation strength)
- Selection: `LogisticRegressionCV` with `TimeSeriesSplit`

### Lasso Logistic Regression

Logistic regression with L1 regularisation. Produces sparse solutions
(feature selection).

```yaml
models:
  lasso:
    C_grid: [0.001, 0.01, 0.1, 1, 10]
    cv_folds: 3
```

- Solver: `liblinear` (supports L1)
- Identical CV procedure to Ridge

### Elastic Net Logistic Regression

Combines L1 and L2 penalties: `alpha * L1 + (1-alpha) * L2`.

```yaml
models:
  elasticnet:
    C_grid: [0.01, 0.1, 1]
    l1_ratio_grid: [0.3, 0.7, 0.9]
    cv_folds: 3
```

- Solver: `saga` (required for Elastic Net)
- Two hyperparameters tuned jointly: `C` and `l1_ratio`

### PCA + Logistic Regression

Dimensionality reduction followed by logistic regression (no penalty).

```yaml
models:
  pca_logistic:
    max_components: null    # null = elbow at 90% explained variance
    cv_folds: 5
```

- PCA is fitted on training data only (no lookahead)
- Number of components = min(max_components, elbow(90% variance))
- Logistic regression on reduced features with no regularisation

---

## Tree models (`src/models/tree_models.py`)

### Random Forest

Ensemble of decision trees with bootstrap sampling and feature subsampling.

```yaml
models:
  random_forest:
    n_estimators_grid: [100, 200]
    max_depth_grid: [3, 5]
    min_samples_leaf_grid: [1, 3]
    cv_folds: 3
    n_jobs: -1
```

- Hyperparameter search: `GridSearchCV` with `TimeSeriesSplit`
- Scoring: `neg_log_loss`
- Out-of-bag probability calibration available but disabled by default

### XGBoost

Gradient boosted trees with the XGBoost library.

```yaml
models:
  xgboost:
    n_estimators: 300
    max_depth_grid: [3, 5]
    min_child_weight_grid: [1]
    eta_grid: [0.1]
    colsample_bytree_grid: [0.8]
    gamma_grid: [0]
    cv_folds: 3
```

- Hyperparameter search: `GridSearchCV` with `TimeSeriesSplit`
- `eval_metric: logloss`
- `n_jobs=-1` within each XGBoost estimator; `n_jobs=1` in GridSearchCV to avoid
  nested parallelism

---

## Neural models (`src/models/neural_models.py`)

Both neural models are implemented in **PyTorch**.

### DNN (Deep Neural Network)

Fully connected feedforward network with batch normalisation.

```yaml
models:
  dnn:
    hidden_layers: [20, 10, 5]   # layer widths (default.yaml)
    activation: "relu"
    epochs: 100
    early_stopping_patience: 10
    learning_rate: 0.001
    l1_reg: 0.0001
    batch_norm: true
    dropout: 0.0
    batch_size: 512
    seed: 42
```

Architecture:
```
Input(n_features) -> Dense(20) -> BN -> ReLU -> Dense(10) -> BN -> ReLU
                  -> Dense(5) -> BN -> ReLU -> Dense(2) -> Softmax
```

### LSTM

Long Short-Term Memory recurrent network that processes a sequence of weekly
feature vectors for each stock.

```yaml
models:
  lstm:
    hidden_units: 30
    seq_len: 8          # weeks of history per sample
    epochs: 100
    early_stopping_patience: 10
    learning_rate: 0.001
    dropout: 0.0
    batch_size: 256
    seed: 42
```

Input shape: `(batch, seq_len=8, n_features)` - uses the most recent 8 weeks
of standardised features for each (date, ticker) observation.

---

## Ensemble (`src/models/ensemble.py`)

A simple probability-average ensemble over all base models included in the run.

```python
ensemble_prob = mean([model.predict_proba(X)[:, 1] for model in fitted_models])
```

- No additional fitting required
- Automatically includes all base models that succeeded in the current window
- The paper shows the ensemble is the top performer across most metrics

---

## Model selection and adding new models

To add a new model:

1. Create a class in `src/models/` inheriting from `BaseStockModel`
2. Implement `fit(X_train, y_train, X_val, y_val)` and `predict_proba(X)`
3. Register it in `src/models/base.py::get_model()`
4. Add a config block in `default.yaml` and `fast.yaml`
5. Add its name to `models.enabled` in the config

---

## Computational notes

| Model | Approx. time per window (fast.yaml) |
|---|---|
| Ridge / Lasso | < 30 s |
| Elastic Net | < 60 s |
| PCA-Logistic | < 30 s |
| Random Forest | 2-5 min |
| XGBoost | 2-5 min |
| DNN | 1-3 min |
| LSTM | 3-8 min |
| **Total (7 windows)** | **~2-4 hours** |

With `fast.yaml` reduced grids, the typical runtime is 45-90 minutes for the
linear/tree models plus ensemble.