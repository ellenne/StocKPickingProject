"""
src/reports/export.py
──────────────────────
Exports all backtest artefacts to the outputs directory:
  - Predictions CSV/parquet
  - Holdings history parquet
  - Metrics CSV
  - Current picks Markdown report
  - Feature availability CSV

Also provides a convenience function ``generate_full_report`` that orchestrates
all charts and table exports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.reports.charts import (
    plot_dnn_training_history,
    plot_drawdown,
    plot_equity_curves,
    plot_feature_importance,
    plot_model_comparison_table,
    plot_rolling_sharpe,
)
from src.reports.tables import (
    build_current_picks_table,
    build_feature_availability_report,
    format_performance_table,
)

logger = logging.getLogger(__name__)


def save_predictions(predictions: pd.DataFrame, outputs_dir: Path) -> None:
    path = outputs_dir / "predictions.parquet"
    predictions.to_parquet(path)
    logger.info("Predictions saved: %s", path)


def save_holdings(holdings: dict[str, pd.DataFrame], outputs_dir: Path) -> None:
    for model_name, df in holdings.items():
        path = outputs_dir / f"holdings_{model_name}.parquet"
        df.to_parquet(path)
    logger.info("Holdings saved for %d models", len(holdings))


def save_metrics(metrics_df: pd.DataFrame, outputs_dir: Path) -> None:
    path = outputs_dir / "metrics.csv"
    metrics_df.to_csv(path)
    logger.info("Metrics saved: %s", path)


def save_equity_curves(returns_dict: dict[str, pd.Series], outputs_dir: Path) -> None:
    df = pd.DataFrame(returns_dict)
    path = outputs_dir / "equity_curves.csv"
    df.to_csv(path)
    logger.info("Equity curves saved: %s", path)


def generate_current_picks_report(
    predictions: pd.DataFrame,
    outputs_dir: Path,
    top_n: int = 50,
    fmt: str = "png",
) -> str:
    """Generate and save the Markdown current picks report.

    Returns the Markdown string.
    """
    prob_cols = sorted([c for c in predictions.columns if c.endswith("_prob")])
    primary_col = "ensemble_prob" if "ensemble_prob" in prob_cols else prob_cols[0]

    picks = build_current_picks_table(
        predictions,
        prob_col=primary_col,
        all_prob_cols=prob_cols,
        top_n=top_n,
    )

    latest_date = predictions.index.get_level_values("date").max().date()

    lines = [
        f"# Current Stock Picks – {latest_date}",
        "",
        f"> Rebalance date: **{latest_date}**  |  Portfolio size: **{top_n}**",
        "> Strategy: Equal-weight, long-only, weekly rebalance",
        "",
        "## Top Picks",
        "",
    ]

    cols_to_show = ["rebalance_date", "ticker"] + prob_cols[:8]
    cols_to_show = [c for c in cols_to_show if c in picks.columns]
    lines.append(picks[cols_to_show].to_markdown())
    lines.append("")

    # Notes
    lines += [
        "## Notes",
        "",
        "- Probabilities represent P(outperform cross-sectional median next week).",
        "- `ensemble_prob` is the simple average of all model probabilities.",
        "- Rankings are based on the most recently trained models.",
        "- **This is for research purposes only – not financial advice.**",
        "",
    ]

    md = "\n".join(lines)
    path = outputs_dir / "current_picks.md"
    path.write_text(md, encoding="utf-8")
    logger.info("Current picks report: %s", path)

    # Also save CSV — fall back to a timestamped name if the file is locked
    picks_path = outputs_dir / "current_picks.csv"
    try:
        picks.to_csv(picks_path)
        logger.info("Current picks CSV: %s", picks_path)
    except PermissionError:
        from datetime import datetime
        fallback = outputs_dir / f"current_picks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        picks.to_csv(fallback)
        logger.warning(
            "current_picks.csv is locked (open in another program?). "
            "Saved to %s instead.", fallback
        )

    return md


def generate_full_report(
    predictions: pd.DataFrame,
    returns_dict: dict[str, pd.Series],
    holdings_dict: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    feature_panel: pd.DataFrame,
    outputs_dir: Path,
    top_n: int = 50,
    fmt: str = "png",
    dnn_histories: dict | None = None,
) -> None:
    """Orchestrate all report generation.

    Parameters
    ----------
    predictions:
        Full out-of-sample predictions DataFrame.
    returns_dict:
        {model_name -> weekly return Series}.
    holdings_dict:
        {model_name -> holdings DataFrame}.
    metrics_df:
        Model metrics comparison table.
    feature_panel:
        The processed feature panel (for feature availability).
    outputs_dir:
        Where to write all outputs.
    top_n:
        Portfolio size.
    fmt:
        Chart file format.
    dnn_histories:
        Optional dict of DNN training histories from run_rolling_backtest.
        When provided, ``dnn_training_curves.<fmt>`` is generated.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Data exports ──────────────────────────────────────────────────────
    save_predictions(predictions, outputs_dir)
    save_holdings(holdings_dict, outputs_dir)
    save_metrics(metrics_df, outputs_dir)
    save_equity_curves(returns_dict, outputs_dir)

    # ── Charts ────────────────────────────────────────────────────────────
    plot_equity_curves(returns_dict, outputs_dir, fmt)

    ensemble_ret = returns_dict.get("ensemble", list(returns_dict.values())[0])
    bench_ret = returns_dict.get("benchmark_1n", ensemble_ret * 0)
    plot_drawdown(ensemble_ret, bench_ret, outputs_dir, fmt)
    plot_rolling_sharpe(returns_dict, outputs_dir, fmt)
    plot_model_comparison_table(metrics_df, outputs_dir, fmt)

    if dnn_histories:
        plot_dnn_training_history(dnn_histories, outputs_dir, fmt)

    # ── Tables ────────────────────────────────────────────────────────────
    build_feature_availability_report(feature_panel, outputs_dir)
    fmt_metrics = format_performance_table(metrics_df)
    fmt_metrics.to_csv(outputs_dir / "metrics_formatted.csv")

    # ── Current picks ─────────────────────────────────────────────────────
    generate_current_picks_report(predictions, outputs_dir, top_n, fmt)

    logger.info("Full report generated in %s", outputs_dir)
