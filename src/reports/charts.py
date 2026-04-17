"""
src/reports/charts.py
──────────────────────
Generates all charts for the backtest report:

1. Equity curves (cumulative log-return) – strategy vs benchmarks
2. Drawdown chart
3. Rolling 12M Sharpe ratio
4. Portfolio size analysis (average return vs N)
5. Feature importance bar chart
6. Model comparison table (styled heatmap)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: Path, fmt: str = "png") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Chart saved: %s", path.with_suffix(f".{fmt}"))


def plot_equity_curves(
    returns_dict: dict[str, pd.Series],
    outputs_dir: Path,
    fmt: str = "png",
    title: str = "Cumulative Returns",
) -> None:
    """Plot cumulative log-return for each series in *returns_dict*.

    Parameters
    ----------
    returns_dict:
        {label → weekly log-return Series}
    outputs_dir:
        Output folder.
    fmt:
        File format (png/pdf/svg).
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    styles = {
        "benchmark_1n": dict(color="black", linestyle="--", linewidth=1.8, label="1/N Benchmark"),
        "spy": dict(color="grey", linestyle=":", linewidth=1.8, label="SPY"),
        "ensemble": dict(color="#e63946", linewidth=2.2, label="Ensemble"),
        "ridge": dict(color="#457b9d", linewidth=1.5),
        "lasso": dict(color="#1d3557", linewidth=1.5),
        "elasticnet": dict(color="#a8dadc", linewidth=1.5),
        "pca_logistic": dict(color="#f4a261", linewidth=1.5),
        "random_forest": dict(color="#2a9d8f", linewidth=1.5),
        "xgboost": dict(color="#e76f51", linewidth=1.5),
        "dnn": dict(color="#8338ec", linewidth=1.5),
        "lstm": dict(color="#3a86ff", linewidth=1.5),
    }

    for label, series in returns_dict.items():
        if series is None or series.empty:
            continue
        cumret = series.cumsum()  # cumulative log return
        kw = styles.get(label, {"linewidth": 1.5})
        kw.setdefault("label", label.upper())
        ax.plot(cumret.index, cumret.values, **kw)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative log-return", fontsize=11)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outputs_dir / "equity_curves", fmt)


def plot_drawdown(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    outputs_dir: Path,
    fmt: str = "png",
    strategy_name: str = "Ensemble",
) -> None:
    """Plot drawdown chart for strategy and benchmark."""
    fig, ax = plt.subplots(figsize=(14, 5))

    def _dd(r):
        cum = (1 + r).cumprod()
        return (cum / cum.cummax()) - 1

    dd_strat = _dd(strategy_returns)
    dd_bench = _dd(benchmark_returns)

    ax.fill_between(dd_strat.index, dd_strat.values, 0, alpha=0.5, color="#e63946", label=strategy_name)
    ax.fill_between(dd_bench.index, dd_bench.values, 0, alpha=0.3, color="grey", label="1/N Benchmark")
    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outputs_dir / "drawdown", fmt)


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    outputs_dir: Path,
    fmt: str = "png",
    window: int = 52,
) -> None:
    """Plot rolling 12M Sharpe ratio for each model."""
    from src.backtest.metrics import rolling_sharpe

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (label, series) in enumerate(returns_dict.items()):
        if series is None or series.empty:
            continue
        rs = rolling_sharpe(series, window)
        ax.plot(rs.index, rs.values, label=label.upper(), color=colors[i % 10], linewidth=1.4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {window}-Week Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, outputs_dir / "rolling_sharpe", fmt)


def plot_feature_importance(
    importances: pd.Series,
    outputs_dir: Path,
    fmt: str = "png",
    top_n: int = 20,
    title: str = "Feature Importances",
) -> None:
    """Horizontal bar chart of feature importances (e.g. from Random Forest)."""
    top = importances.abs().nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    top.plot.barh(ax=ax, color="#457b9d")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, outputs_dir / "feature_importance", fmt)


def plot_model_comparison_table(
    metrics_df: pd.DataFrame,
    outputs_dir: Path,
    fmt: str = "png",
) -> None:
    """Render the metrics comparison table as a matplotlib figure."""
    display_cols = [
        "ann_return_gross", "ann_volatility", "sharpe_gross",
        "max_drawdown", "hit_rate", "ann_turnover", "btc_bps",
    ]
    col_labels = {
        "ann_return_gross": "Return p.a.",
        "ann_volatility": "Vol p.a.",
        "sharpe_gross": "Sharpe",
        "max_drawdown": "Max DD",
        "hit_rate": "Hit Rate",
        "ann_turnover": "Turnover",
        "btc_bps": "BTC (bps)",
    }
    cols = [c for c in display_cols if c in metrics_df.columns]
    df = metrics_df[cols].rename(columns=col_labels)

    fig, ax = plt.subplots(figsize=(len(cols) * 1.5 + 2, len(df) * 0.55 + 1.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.apply(lambda col: col.map(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))).values,
        rowLabels=df.index.tolist(),
        colLabels=df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, outputs_dir / "model_comparison_table", fmt)
