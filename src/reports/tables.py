"""
src/reports/tables.py
──────────────────────
Generates formatted table outputs:

  - Performance summary table (mirrors Table 4 in the paper)
  - Feature availability table
  - Confusion matrix table
  - Current picks table with probabilities and optional features
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_performance_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready performance table (percentage-formatted)."""
    pct_cols = [
        "ann_return_gross", "ann_return_net", "ann_volatility",
        "max_drawdown", "hit_rate", "bench_1n_ann_return", "spy_ann_return",
    ]
    ratio_cols = [
        "sharpe_gross", "sharpe_net", "capm_beta", "bench_1n_sharpe", "spy_sharpe",
    ]
    bps_cols = ["btc_bps"]
    int_cols = ["ann_turnover"]

    out = metrics_df.copy()
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.1%}" if isinstance(x, float) else str(x))
    for col in ratio_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))
    for col in bps_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.1f}" if isinstance(x, float) else str(x))
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.1f}" if isinstance(x, float) else str(x))
    return out


def build_current_picks_table(
    predictions: pd.DataFrame,
    prob_col: str = "ensemble_prob",
    all_prob_cols: list[str] | None = None,
    top_n: int = 50,
    feature_panel: pd.DataFrame | None = None,
    linear_coef: dict | None = None,
) -> pd.DataFrame:
    """Build the current top-N picks table for the latest available date.

    Parameters
    ----------
    predictions:
        MultiIndex (date, ticker) predictions DataFrame.
    prob_col:
        Primary ranking column (default: ensemble_prob).
    all_prob_cols:
        All per-model probability columns to include in output.
    top_n:
        Number of stocks to show.
    feature_panel:
        If provided, the latest feature values are joined to the output.
    linear_coef:
        {model_name: coef_array} for optional top-feature display.

    Returns
    -------
    pd.DataFrame
        Ranked top-N picks with probabilities.
    """
    latest_date = predictions.index.get_level_values("date").max()
    day_pred = predictions.xs(latest_date, level="date")

    if prob_col not in day_pred.columns:
        # Fallback to first available prob column
        prob_cols_available = [c for c in day_pred.columns if c.endswith("_prob")]
        if not prob_cols_available:
            raise ValueError("No probability columns found in predictions.")
        prob_col = prob_cols_available[0]
        logger.warning("Falling back to prob column: %s", prob_col)

    ranked = (
        day_pred[prob_col]
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    ranked.columns = ["ticker", "ensemble_prob"]
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.insert(1, "rebalance_date", latest_date.date())

    # Add individual model probabilities
    if all_prob_cols:
        for col in all_prob_cols:
            if col in day_pred.columns:
                ranked[col] = day_pred[col].reindex(ranked["ticker"]).values

    # Optionally join latest feature values
    if feature_panel is not None and not feature_panel.empty:
        try:
            feat = feature_panel.xs(latest_date, level="date")
            ranked = ranked.merge(
                feat.reset_index(),
                on="ticker",
                how="left",
            )
        except (KeyError, Exception):
            pass

    ranked = ranked.set_index("rank")
    return ranked


def build_feature_availability_report(
    panel: pd.DataFrame,
    outputs_dir=None,
) -> pd.DataFrame:
    """Build and optionally save the feature availability report."""
    from src.features.fundamentals import feature_availability_report

    report = feature_availability_report(panel)
    if outputs_dir is not None:
        path = outputs_dir / "feature_availability.csv"
        report.to_csv(path, index=False)
        logger.info("Feature availability report saved: %s", path)
    return report


def confusion_matrix_stats(
    predictions: pd.DataFrame,
    prob_col: str,
    target_col: str = "target",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute confusion matrix statistics."""
    df = predictions[[prob_col, target_col]].dropna()
    pred = (df[prob_col] >= threshold).astype(int)
    actual = df[target_col].astype(int)

    tp = ((pred == 1) & (actual == 1)).sum()
    fp = ((pred == 1) & (actual == 0)).sum()
    tn = ((pred == 0) & (actual == 0)).sum()
    fn = ((pred == 0) & (actual == 1)).sum()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)  # recall for positives
    specificity = tn / max(tn + fp, 1)  # recall for negatives
    precision = tp / max(tp + fp, 1)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
