"""
src/backtest/rolling_training.py
─────────────────────────────────
Orchestrates the rolling / expanding-window training and test loop described
in the paper (Figure 1):

  Train on 3 years (156 weeks) → test on next year (52 weeks) → retrain.

For each annual window:
  1. Slice the feature matrix and labels into train/val/test splits.
  2. Build a fresh ``FeaturePreprocessor`` fitted *only on training data*.
  3. Fit each enabled model.
  4. Predict outperformance probabilities on the test weeks.
  5. Accumulate predictions and store them.

Returns a DataFrame of out-of-sample predictions with columns:
  [model_name, prob, date, ticker]

Anti-lookahead guarantees
──────────────────────────
* Preprocessing (mean/std, winsorise bounds) is computed only on training rows.
* The target forward-return is NOT included in the feature matrix.
* Model selection (hyperparameter CV) uses only the training period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import Config
from src.features.preprocessing import FeaturePreprocessor, build_feature_matrix
from src.models.base import BaseStockModel, get_model
from src.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)

WEEKS_PER_YEAR = 52


@dataclass
class TrainTestWindow:
    """One rolling window's split indices."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    val_start: pd.Timestamp   # last 20% of training period


def _generate_windows(
    all_dates: pd.DatetimeIndex,
    train_years: int,
    test_years: int,
    val_fraction: float,
) -> list[TrainTestWindow]:
    """Generate rolling train/test date windows.

    The first training period starts from ``all_dates[0]`` and covers
    ``train_years`` of data.  Each subsequent window slides forward by
    ``test_years``.
    """
    windows: list[TrainTestWindow] = []
    train_weeks = train_years * WEEKS_PER_YEAR
    test_weeks = test_years * WEEKS_PER_YEAR

    idx = 0
    while idx + train_weeks + test_weeks <= len(all_dates):
        train_dates = all_dates[idx : idx + train_weeks]
        test_dates = all_dates[idx + train_weeks : idx + train_weeks + test_weeks]

        n_val = max(1, int(len(train_dates) * val_fraction))
        val_start = train_dates[len(train_dates) - n_val]

        windows.append(
            TrainTestWindow(
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                val_start=val_start,
            )
        )
        idx += test_weeks  # slide by one test year (re-use last train data)

    logger.info(
        "Generated %d rolling windows (train=%dy, test=%dy)",
        len(windows),
        train_years,
        test_years,
    )
    return windows


def run_rolling_backtest(
    feature_panel: pd.DataFrame,
    target_series: pd.Series,
    fwd_return_series: pd.Series,
    cfg: Config,
    models_override: list[str] | None = None,
) -> pd.DataFrame:
    """Execute the full rolling training / prediction loop.

    Parameters
    ----------
    feature_panel:
        MultiIndex (date, ticker) DataFrame of feature values
        (output of ``build_feature_matrix``).
    target_series:
        MultiIndex (date, ticker) Series of binary labels.
    fwd_return_series:
        MultiIndex (date, ticker) Series of forward log-returns.
    cfg:
        Project config.
    models_override:
        If provided, use this list of model names instead of cfg.enabled_models.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - predictions: Out-of-sample predictions with columns
          [date, ticker, fwd_return, target, <model>_prob, ensemble_prob]
        - dnn_histories: dict mapping window_index (1-based) to a list of
          per-epoch dicts {epoch, train_loss, val_loss, is_best}.
          Empty dict when DNN is not in the enabled model list.
    """
    enabled = models_override or cfg.enabled_models
    # Exclude "ensemble" – it's built from the others
    base_model_names = [m for m in enabled if m != "ensemble"]
    include_ensemble = "ensemble" in enabled

    all_dates = feature_panel.index.get_level_values("date").unique().sort_values()
    windows = _generate_windows(
        all_dates,
        cfg.train_years,
        cfg.test_years,
        cfg._raw["rolling"]["val_fraction"],
    )

    if not windows:
        raise ValueError(
            "Not enough data to form even one rolling window. "
            f"Need at least {cfg.train_years + cfg.test_years} years of data."
        )

    all_preds: list[pd.DataFrame] = []
    # key: 1-based window index, value: list of per-epoch history dicts
    dnn_histories: dict[int, list[dict]] = {}

    for w_idx, window in enumerate(windows):
        w_num = w_idx + 1
        n_windows = len(windows)
        logger.info(
            "── Window %d/%d │ train %s->%s │ test %s->%s",
            w_num, n_windows,
            window.train_start.date(), window.train_end.date(),
            window.test_start.date(), window.test_end.date(),
        )

        # ── Slice data ────────────────────────────────────────────────────
        dates = feature_panel.index.get_level_values("date")

        train_mask = (dates >= window.train_start) & (dates <= window.train_end)
        val_mask   = (dates >= window.val_start)   & (dates <= window.train_end)
        test_mask  = (dates >= window.test_start)  & (dates <= window.test_end)

        X_all_train = feature_panel.loc[train_mask]
        y_all_train = target_series.loc[train_mask]
        X_val       = feature_panel.loc[val_mask]
        y_val       = target_series.loc[val_mask]
        X_test      = feature_panel.loc[test_mask]

        # Drop rows with NaN targets
        valid_train = y_all_train.notna()
        X_tr = X_all_train.loc[valid_train]
        y_tr = y_all_train.loc[valid_train]

        valid_val = y_val.notna()
        X_v = X_val.loc[valid_val]
        y_v = y_val.loc[valid_val]

        if len(X_tr) < 100:
            logger.warning("Window %d: too few training rows (%d), skipping.", w_num, len(X_tr))
            continue

        # ── Preprocessing (fit on train only) ─────────────────────────────
        prep = FeaturePreprocessor(
            winsorize=cfg._raw["features"]["winsorize"],
            winsorize_pct=cfg._raw["features"]["winsorize_pct"],
        )
        prep.fit(X_tr)
        X_tr_s   = prep.transform(X_tr)
        X_v_s    = prep.transform(X_v)
        X_test_s = prep.transform(X_test)

        y_tr_np = y_tr.values
        y_v_np  = y_v.values if len(y_v) > 0 else None

        # ── Train and predict each base model ─────────────────────────────
        fitted_models: list[BaseStockModel] = []
        test_probas: dict[str, np.ndarray] = {}

        for model_name in base_model_names:
            try:
                model = get_model(model_name, cfg)

                # Tag DNN/LSTM with window context for progress logging
                if hasattr(model, "_window_tag"):
                    model._window_tag = (
                        f"{model_name} W{w_num}/{n_windows} "
                        f"({window.train_start.year}-{window.test_end.year})"
                    )

                model.fit(
                    X_tr_s,
                    y_tr_np,
                    X_v_s if len(X_v_s) > 0 else None,
                    y_v_np,
                )
                proba = model.predict_proba(X_test_s)[:, 1]
                test_probas[model_name] = proba
                fitted_models.append(model)

                # Collect DNN training history for loss-curve visualisation
                if model_name == "dnn" and hasattr(model, "training_history_"):
                    dnn_histories[w_num] = {
                        "history":     model.training_history_,
                        "train_start": str(window.train_start.date()),
                        "train_end":   str(window.train_end.date()),
                        "test_start":  str(window.test_start.date()),
                        "test_end":    str(window.test_end.date()),
                    }

            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] failed in window %d: %s", model_name, w_num, exc)

        # ── Ensemble ──────────────────────────────────────────────────────
        if include_ensemble and fitted_models:
            ensemble = EnsembleModel(fitted_models)
            ens_proba = ensemble.predict_proba(X_test_s)[:, 1]
            test_probas["ensemble"] = ens_proba

        # ── Assemble predictions DataFrame ────────────────────────────────
        pred_df = pd.DataFrame(index=X_test.index)
        pred_df["fwd_return"] = fwd_return_series.reindex(X_test.index)
        pred_df["target"]     = target_series.reindex(X_test.index)

        for model_name, proba in test_probas.items():
            if len(proba) == len(pred_df):
                pred_df[f"{model_name}_prob"] = proba
            else:
                logger.warning(
                    "Probability length mismatch for %s in window %d "
                    "(expected %d, got %d).",
                    model_name, w_num, len(pred_df), len(proba),
                )

        all_preds.append(pred_df)

    if not all_preds:
        raise RuntimeError("Rolling backtest produced no predictions.")

    result = pd.concat(all_preds).sort_index()
    logger.info(
        "Rolling backtest complete: %d out-of-sample rows, date range %s - %s",
        len(result),
        result.index.get_level_values("date").min().date(),
        result.index.get_level_values("date").max().date(),
    )
    return result, dnn_histories
