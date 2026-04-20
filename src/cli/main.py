"""
src/cli/main.py
───────────────
Command-line interface for the StockPickingML project.

Commands
────────
  download-data     Download and cache raw price/fundamental data.
  build-dataset     Assemble the processed weekly panel.
  train-backtest    Run the rolling training + backtest loop.
  current-picks     Output the latest top-N stock picks.
  full-run          Execute the complete pipeline end-to-end.

Usage examples
--------------
  python -m src.cli.main download-data
  python -m src.cli.main build-dataset
  python -m src.cli.main train-backtest --models ridge,lasso,ensemble
  python -m src.cli.main current-picks --top-n 50
  python -m src.cli.main full-run --config configs/default.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--config", default="configs/default.yaml", show_default=True,
              help="Path to YAML config file.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Debug logging.")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """StockPickingML – ML-based stock picker (Wolff & Echterling 2022)."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    from src.config import load_config
    ctx.obj["cfg"] = load_config(config)
    ctx.obj["verbose"] = verbose


# ─────────────────────────────────────────────────────────────────────────────
#  download-data
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("download-data")
@click.option("--force-refresh", is_flag=True, default=False,
              help="Re-download even if cache files exist.")
@click.pass_context
def download_data(ctx: click.Context, force_refresh: bool) -> None:
    """Download and cache price, fundamental, and sector data."""
    cfg = ctx.obj["cfg"]
    console.rule("[bold blue]Downloading Data")

    from src.data.universe import Universe
    from src.data.prices import download_prices
    from src.data.fundamentals import download_fundamentals

    data_cfg = cfg._raw["data"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"] or __import__("pandas").Timestamp.today().strftime("%Y-%m-%d")

    console.print(f"Universe source: {cfg._raw['universe']['source']}")
    console.print(f"Date range: {start} to {end}")

    universe = Universe(cfg)
    tickers = universe.all_tickers
    console.print(f"Tickers: [bold]{len(tickers)}[/bold]")

    console.print("\n[bold]Downloading prices …[/bold]")
    download_prices(tickers, start, end, cfg.cache_dir, force_refresh)

    console.print("\n[bold]Downloading fundamentals …[/bold]")
    download_fundamentals(tickers, cfg.cache_dir, force_refresh)

    console.print("\n[green]✓ Data download complete.[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  build-dataset
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("build-dataset")
@click.option("--force-refresh", is_flag=True, default=False)
@click.pass_context
def build_dataset_cmd(ctx: click.Context, force_refresh: bool) -> None:
    """Assemble the processed weekly panel (prices + features + target)."""
    import pandas as pd
    cfg = ctx.obj["cfg"]
    console.rule("[bold blue]Building Dataset")

    from src.data.loaders import build_dataset
    from src.features.technical import compute_technical_features
    from src.features.preprocessing import build_feature_matrix
    from src.features.target import add_target, forward_return_col
    from src.backtest.portfolio import download_spy_returns

    console.print("Assembling raw panel …")
    panel = build_dataset(cfg, force_refresh)
    console.print(f"Raw panel shape: {panel.shape}")

    # ── Technical features ────────────────────────────────────────────────
    console.print("Computing technical features …")
    close_panel = panel["close"].unstack("ticker")
    vol_panel = panel["volume"].unstack("ticker") if "volume" in panel.columns else None

    data_cfg = cfg._raw["data"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    spy_ret = download_spy_returns(start, end, cfg.cache_dir)
    tech_features = compute_technical_features(
        close_panel,
        vol_panel if vol_panel is not None else close_panel * 0,
        index_return_series=spy_ret,
    )
    panel = panel.join(tech_features, how="left")

    # ── Target ────────────────────────────────────────────────────────────
    console.print("Computing binary target …")
    strict_gt = cfg._raw["target"]["label_strict_gt"]
    panel = add_target(panel, strict_gt=strict_gt)
    panel = forward_return_col(panel)

    # ── Persist ───────────────────────────────────────────────────────────
    out_path = cfg.processed_dir / "features_panel.parquet"
    panel.to_parquet(out_path)
    console.print(f"[green]✓ Features panel saved → {out_path}  (shape {panel.shape})[/green]")

    # ── Feature availability report ───────────────────────────────────────
    from src.reports.tables import build_feature_availability_report
    report = build_feature_availability_report(panel, cfg.outputs_dir)
    available = report[report["status"] == "available"]
    missing = report[report["status"] == "missing"]
    console.print(
        f"Features: [green]{len(available)} available[/green] / "
        f"[red]{len(missing)} missing[/red]"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  train-backtest
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("train-backtest")
@click.option("--models", default=None,
              help="Comma-separated model names. Default: all enabled in config.")
@click.option("--top-n", default=None, type=int,
              help="Portfolio size (overrides config).")
@click.option("--no-tc", is_flag=True, default=False,
              help="Disable transaction cost deduction.")
@click.pass_context
def train_backtest(ctx: click.Context, models: str | None, top_n: int | None, no_tc: bool) -> None:
    """Run the rolling training + backtest loop and generate reports."""
    import pandas as pd
    cfg = ctx.obj["cfg"]
    console.rule("[bold blue]Training & Backtesting")

    # ── Load feature panel ────────────────────────────────────────────────
    feat_path = cfg.processed_dir / "features_panel.parquet"
    if not feat_path.exists():
        console.print(
            "[red]Feature panel not found. Run [bold]build-dataset[/bold] first.[/red]"
        )
        sys.exit(1)

    console.print(f"Loading feature panel from {feat_path} …")
    panel = pd.read_parquet(feat_path)

    # ── Build clean feature matrix ────────────────────────────────────────
    from src.features.preprocessing import build_feature_matrix

    console.print("Building feature matrix …")
    feat_cfg = cfg._raw["features"]
    feature_panel = build_feature_matrix(
        panel,
        ffill_limit=feat_cfg["ffill_limit"],
        include_technical=feat_cfg["technical"],
        include_fundamental=feat_cfg["fundamental"],
        include_sector_dummies=feat_cfg["sector_dummies"],
    )

    target_series = panel["target"]
    fwd_return_series = panel.get("fwd_return", panel.get("log_return", None))
    if fwd_return_series is None:
        console.print("[red]fwd_return column missing from panel.[/red]")
        sys.exit(1)

    # ── Model selection ───────────────────────────────────────────────────
    models_list = models.split(",") if models else None

    # ── Rolling backtest ──────────────────────────────────────────────────
    from src.backtest.rolling_training import run_rolling_backtest

    console.print("Starting rolling backtest …")
    predictions, dnn_histories = run_rolling_backtest(
        feature_panel, target_series, fwd_return_series, cfg, models_list
    )
    if dnn_histories:
        import json
        dnn_hist_path = cfg.outputs_dir / "dnn_training_histories.json"
        dnn_hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dnn_hist_path, "w", encoding="utf-8") as _f:
            json.dump(dnn_histories, _f, indent=2)
        console.print(f"  DNN training histories saved to {dnn_hist_path}")

    # ── Portfolio construction ────────────────────────────────────────────
    from src.backtest.portfolio import (
        build_benchmark_returns,
        build_portfolio_returns,
        download_spy_returns,
    )
    from src.backtest.metrics import build_metrics_table, compute_all_metrics

    n = top_n or cfg.top_n
    prob_cols = sorted([c for c in predictions.columns if c.endswith("_prob")])
    if not prob_cols:
        console.print("[red]No probability columns in predictions.[/red]")
        sys.exit(1)

    data_cfg = cfg._raw["data"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    spy_ret = download_spy_returns(start, end, cfg.cache_dir)
    bench_ret = build_benchmark_returns(predictions)

    returns_dict: dict[str, pd.Series] = {
        "benchmark_1n": bench_ret,
        "spy": spy_ret,
    }
    holdings_dict: dict[str, pd.DataFrame] = {}
    all_metrics: dict[str, dict] = {}

    tc_cfg = cfg._raw["transaction_costs"]
    tc_enabled = tc_cfg["enabled"] and not no_tc
    one_way_bps = tc_cfg["one_way_bps"]

    for prob_col in prob_cols:
        model_name = prob_col.replace("_prob", "")
        console.print(f"  Portfolio [{model_name}] top-{n} …")
        port_ret, holdings = build_portfolio_returns(
            predictions, "fwd_return", prob_col, n
        )
        returns_dict[model_name] = port_ret
        holdings_dict[model_name] = holdings

        metrics = compute_all_metrics(
            port_ret, bench_ret, spy_ret, predictions, prob_col,
            holdings, one_way_bps, tc_enabled,
        )
        all_metrics[model_name] = metrics

    metrics_df = build_metrics_table(all_metrics)

    # ── Reports ───────────────────────────────────────────────────────────
    from src.reports.export import generate_full_report

    generate_full_report(
        predictions, returns_dict, holdings_dict, metrics_df,
        feature_panel, cfg.outputs_dir, n,
        fmt=cfg._raw["outputs"]["chart_format"],
        dnn_histories=dnn_histories if dnn_histories else None,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    from src.reports.tables import format_performance_table

    console.rule("[bold green]Backtest Results")
    fmt_df = format_performance_table(metrics_df)
    _print_table(fmt_df[["ann_return_gross", "ann_volatility", "sharpe_gross",
                           "max_drawdown", "hit_rate", "btc_bps"]])


def _print_table(df: "pd.DataFrame") -> None:
    """Print a pandas DataFrame as a Rich table."""
    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Model", style="bold")
    for col in df.columns:
        tbl.add_column(col)
    for idx, row in df.iterrows():
        tbl.add_row(str(idx), *[str(v) for v in row.values])
    console.print(tbl)


# ─────────────────────────────────────────────────────────────────────────────
#  current-picks
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("current-picks")
@click.option("--top-n", default=50, show_default=True, help="Number of stocks to show.")
@click.option("--model", default="ensemble", show_default=True,
              help="Model to rank by (e.g. ensemble, ridge, xgboost).")
@click.pass_context
def current_picks(ctx: click.Context, top_n: int, model: str) -> None:
    """Display and save the latest top-N stock picks."""
    import pandas as pd
    cfg = ctx.obj["cfg"]
    console.rule(f"[bold blue]Current Picks – Top {top_n}")

    pred_path = cfg.outputs_dir / "predictions.parquet"
    if not pred_path.exists():
        console.print(
            "[red]Predictions not found. Run [bold]train-backtest[/bold] first.[/red]"
        )
        sys.exit(1)

    predictions = pd.read_parquet(pred_path)
    prob_col = f"{model}_prob"
    if prob_col not in predictions.columns:
        available = [c for c in predictions.columns if c.endswith("_prob")]
        console.print(f"[yellow]Model '{model}' not found. Available: {available}[/yellow]")
        prob_col = available[0] if available else None
        if prob_col is None:
            sys.exit(1)

    from src.reports.tables import build_current_picks_table
    from src.reports.export import generate_current_picks_report

    all_prob_cols = [c for c in predictions.columns if c.endswith("_prob")]
    picks = build_current_picks_table(
        predictions, prob_col=prob_col, all_prob_cols=all_prob_cols, top_n=top_n
    )

    console.print(
        f"\nLatest rebalance date: [bold]{picks['rebalance_date'].iloc[0]}[/bold]\n"
    )
    tbl = Table(title=f"Top {top_n} Picks ({model.upper()})", show_header=True,
                header_style="bold green")
    display_cols = ["ticker"] + all_prob_cols[:5]
    display_cols = [c for c in display_cols if c in picks.columns]
    for col in display_cols:
        tbl.add_column(col)
    for _, row in picks[display_cols].iterrows():
        tbl.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values])
    console.print(tbl)

    generate_current_picks_report(predictions, cfg.outputs_dir, top_n)
    console.print(f"\n[green]✓ Report saved to {cfg.outputs_dir / 'current_picks.md'}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  compute-shap
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("compute-shap")
@click.option("--panel", default=None, show_default=True,
              help="Path to processed panel parquet. Defaults to data/processed/weekly_panel.parquet.")
@click.option("--models", default=None, show_default=True,
              help="Comma-separated model names. Defaults to enabled models in config.")
@click.option("--window", default=-1, show_default=True, type=int,
              help="Rolling window index (1-based). Use -1 for the last window.")
@click.option("--n-bg", default=300, show_default=True,
              help="Background samples for LinearExplainer / KernelExplainer.")
@click.option("--n-bg-deep", default=100, show_default=True,
              help="Background samples for DeepExplainer (DNN).")
@click.option("--n-test", default=500, show_default=True,
              help="Max test rows for SHAP evaluation (slow explainers).")
@click.option("--nsamples-kernel", default=150, show_default=True,
              help="Coalition samples per test row for KernelExplainer.")
@click.option("--top-n", default=20, show_default=True,
              help="Number of top features shown in bar charts.")
@click.pass_context
def compute_shap_cmd(
    ctx: click.Context,
    panel: str | None,
    models: str | None,
    window: int,
    n_bg: int,
    n_bg_deep: int,
    n_test: int,
    nsamples_kernel: int,
    top_n: int,
) -> None:
    """Compute SHAP feature importances and reproduce paper Figure 8.

    Refits all enabled models on the specified rolling window's training data,
    then runs the appropriate SHAP explainer for each model type:

      Ridge / Lasso / ElasticNet  - LinearExplainer  (fast)\n
      RandomForest / XGBoost      - TreeExplainer    (fast)\n
      DNN                         - DeepExplainer    (fast, PyTorch)\n
      LSTM / PCA-Logistic         - KernelExplainer  (slow, model-agnostic)\n

    Outputs saved to <outputs_dir>/shap/:
      shap_heatmap.<fmt>           - Figure 8 style heatmap\n
      shap_summary_bars.<fmt>      - Top-N features per model\n
      shap_group_comparison.<fmt>  - Technical vs Fundamental fractions\n
      shap_beeswarm_<model>.<fmt>  - Per-model beeswarm plots\n
      shap_importance.csv          - Raw mean |SHAP| table\n
    """
    import numpy as np
    import pandas as pd

    cfg = ctx.obj["cfg"]
    console.rule("[bold blue]SHAP Feature Importance Analysis")

    # ── Load panel ────────────────────────────────────────────────────────
    panel_path = Path(panel) if panel else cfg.cache_dir / "processed" / "weekly_panel.parquet"
    if not panel_path.exists():
        # Try the default processed path
        panel_path = Path("data/processed/weekly_panel.parquet")
    if not panel_path.exists():
        console.print(
            "[red]Processed panel not found. Run [bold]build-dataset[/bold] first.[/red]"
        )
        sys.exit(1)

    console.print(f"Loading panel from {panel_path} …")
    full_panel = pd.read_parquet(panel_path)
    console.print(f"  Panel shape: {full_panel.shape}")

    # ── Separate features from target/return columns ───────────────────
    exclude = {"target", "fwd_return", "fwd_log_return"}
    feature_cols = [c for c in full_panel.columns if c not in exclude]
    feature_panel = full_panel[feature_cols]

    if "target" not in full_panel.columns:
        console.print("[red]'target' column missing from panel.[/red]")
        sys.exit(1)
    target_series = full_panel["target"]

    # ── Generate windows, pick the requested one ───────────────────────
    from src.backtest.rolling_training import _generate_windows

    all_dates = feature_panel.index.get_level_values("date").unique().sort_values()
    windows = _generate_windows(
        all_dates, cfg.train_years, cfg.test_years,
        cfg._raw["rolling"]["val_fraction"],
    )
    if not windows:
        console.print("[red]Not enough data to form a rolling window.[/red]")
        sys.exit(1)

    w_idx = (window - 1) if window > 0 else (len(windows) + window)
    w_idx = max(0, min(w_idx, len(windows) - 1))
    win = windows[w_idx]
    console.print(
        f"  Using window {w_idx + 1}/{len(windows)}: "
        f"train {win.train_start.date()} to {win.train_end.date()}, "
        f"test  {win.test_start.date()} to {win.test_end.date()}"
    )

    # ── Slice train/test ───────────────────────────────────────────────
    dates = feature_panel.index.get_level_values("date")
    train_mask = (dates >= win.train_start) & (dates <= win.train_end)
    test_mask  = (dates >= win.test_start)  & (dates <= win.test_end)

    X_tr_raw = feature_panel.loc[train_mask]
    y_tr_raw = target_series.loc[train_mask]
    X_te_raw = feature_panel.loc[test_mask]

    valid_train = y_tr_raw.notna()
    X_tr_raw = X_tr_raw.loc[valid_train]
    y_tr_raw = y_tr_raw.loc[valid_train]

    # ── Preprocess (fit on train only) ─────────────────────────────────
    from src.features.preprocessing import FeaturePreprocessor

    prep = FeaturePreprocessor(
        winsorize=cfg._raw["features"]["winsorize"],
        winsorize_pct=cfg._raw["features"]["winsorize_pct"],
    )
    prep.fit(X_tr_raw)
    X_tr = prep.transform(X_tr_raw).values.astype(np.float32)
    X_te = prep.transform(X_te_raw).values.astype(np.float32)
    y_tr = y_tr_raw.values.astype(np.int64)
    feature_names = list(feature_cols)

    console.print(
        f"  Train rows: {len(X_tr)}, Test rows: {len(X_te)}, "
        f"Features: {len(feature_names)}"
    )

    # ── Fit models ─────────────────────────────────────────────────────
    model_names = models.split(",") if models else [
        m for m in cfg.enabled_models if m != "ensemble"
    ]
    from src.models.base import get_model

    fitted: dict = {}
    for model_name in model_names:
        console.print(f"  Fitting [bold]{model_name}[/bold] …")
        try:
            m = get_model(model_name, cfg)
            # Pass DataFrame so LSTM gets proper MultiIndex sequences
            m.fit(X_tr_raw.pipe(prep.transform), y_tr,
                  X_te_raw.pipe(prep.transform), None)
            fitted[model_name] = m
        except Exception as exc:
            console.print(f"  [yellow]Warning: {model_name} failed: {exc}[/yellow]")

    if not fitted:
        console.print("[red]No models fitted successfully.[/red]")
        sys.exit(1)

    # ── Run SHAP ───────────────────────────────────────────────────────
    from src.reports.shap_analysis import run_shap_analysis

    fmt = cfg._raw["outputs"]["chart_format"]
    console.print(
        f"\nRunning SHAP for {list(fitted.keys())} "
        f"(n_bg={n_bg}, n_test={n_test}, kernel_samples={nsamples_kernel}) …"
    )
    shap_dict = run_shap_analysis(
        fitted, X_tr, X_te, feature_names,
        outputs_dir=cfg.outputs_dir,
        fmt=fmt,
        n_bg=n_bg,
        n_test=n_test,
        n_bg_deep=n_bg_deep,
        nsamples_kernel=nsamples_kernel,
        top_n_bars=top_n,
    )

    # ── Summary table ──────────────────────────────────────────────────
    if shap_dict:
        from src.reports.shap_analysis import _mean_abs, feature_group
        console.rule("[bold green]Top-5 Features per Model")
        for model_name, sv in shap_dict.items():
            imp   = _mean_abs(sv)
            order = np.argsort(imp)[::-1][:5]
            top   = [(feature_names[i], imp[i], feature_group(feature_names[i]))
                     for i in order]
            tbl_str = "  ".join(f"{n} ({g[0]}): {v:.4f}" for n, v, g in top)
            console.print(f"[bold]{model_name:15s}[/bold]  {tbl_str}")

    shap_dir = cfg.outputs_dir / "shap"
    console.print(
        f"\n[green]SHAP analysis complete. Outputs in {shap_dir}[/green]"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  full-run
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("full-run")
@click.option("--force-refresh", is_flag=True, default=False)
@click.option("--models", default=None, help="Comma-separated model names.")
@click.option("--top-n", default=None, type=int)
@click.pass_context
def full_run(ctx: click.Context, force_refresh: bool, models: str | None, top_n: int | None) -> None:
    """Run the complete pipeline: download → build → train → report."""
    console.rule("[bold magenta]Full Pipeline Run")
    ctx.invoke(download_data, force_refresh=force_refresh)
    ctx.invoke(build_dataset_cmd, force_refresh=force_refresh)
    ctx.invoke(train_backtest, models=models, top_n=top_n, no_tc=False)
    ctx.invoke(current_picks, top_n=top_n or 50, model="ensemble")
    console.rule("[bold green]Pipeline Complete")


if __name__ == "__main__":
    cli()
