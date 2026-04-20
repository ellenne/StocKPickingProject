"""
Smoke-tests for src/reports/shap_analysis.py.

Runs all explainer types on small synthetic datasets so the full
compute_all_shap + plotting pipeline can be verified quickly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


N_SAMPLES = 60
N_FEAT    = 10
FEAT_NAMES = (
    ["pe_ratio", "pb_ratio", "roe", "profit_margin"]     # Fundamental
    + ["mom_4w", "rsi_14", "vol_4w"]                      # Technical
    + ["sector_tech", "sector_fin", "sector_health"]      # Sector
)


def _make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((N_SAMPLES, N_FEAT)).astype(np.float32)
    y   = rng.integers(0, 2, N_SAMPLES).astype(np.int64)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  feature_group
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_group():
    from src.reports.shap_analysis import feature_group, feature_groups
    assert feature_group("pe_ratio")     == "Fundamental"
    assert feature_group("mom_4w")       == "Technical"
    assert feature_group("rsi_14")       == "Technical"
    assert feature_group("sector_tech")  == "Sector"

    groups = feature_groups(FEAT_NAMES)
    assert "pe_ratio" in groups["Fundamental"]
    assert "mom_4w"   in groups["Technical"]
    assert "sector_tech" in groups["Sector"]


# ─────────────────────────────────────────────────────────────────────────────
#  Linear explainer (mocked Ridge)
# ─────────────────────────────────────────────────────────────────────────────

def test_shap_linear():
    from sklearn.linear_model import LogisticRegression
    from src.reports.shap_analysis import _shap_linear

    X, y = _make_data()
    clf = LogisticRegression(max_iter=200, random_state=0).fit(X, y)

    class FakeLinear:
        _clf = clf

    sv = _shap_linear(FakeLinear(), X[:30], X[30:])
    assert sv.shape == (30, N_FEAT), f"bad shape: {sv.shape}"
    assert np.isfinite(sv).all()


# ─────────────────────────────────────────────────────────────────────────────
#  Tree explainer (RandomForest)
# ─────────────────────────────────────────────────────────────────────────────

def test_shap_tree_rf():
    from sklearn.ensemble import RandomForestClassifier
    from src.reports.shap_analysis import _shap_tree

    X, y = _make_data()
    clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    class FakeRF:
        _clf = clf

    sv = _shap_tree(FakeRF(), X[30:])
    assert sv.shape == (30, N_FEAT), f"bad shape: {sv.shape}"
    assert np.isfinite(sv).all()


# ─────────────────────────────────────────────────────────────────────────────
#  Kernel explainer (generic black-box)
# ─────────────────────────────────────────────────────────────────────────────

def test_shap_kernel():
    from sklearn.linear_model import LogisticRegression
    from src.reports.shap_analysis import _shap_kernel

    X, y = _make_data()
    clf = LogisticRegression(max_iter=200, random_state=0).fit(X, y)

    sv = _shap_kernel(
        lambda x: clf.predict_proba(x)[:, 1],
        X[:30], X[30:40],
        n_bg_samples=20, n_test_samples=10, nsamples_kernel=50,
    )
    assert sv.shape == (10, N_FEAT), f"bad shape: {sv.shape}"
    assert np.isfinite(sv).all()


# ─────────────────────────────────────────────────────────────────────────────
#  compute_all_shap + full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_all_shap_and_plots(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from src.reports.shap_analysis import (
        compute_all_shap, plot_shap_heatmap, plot_shap_summary_bars,
        plot_shap_group_comparison, save_shap_csv,
    )

    X, y = _make_data()
    X_bg, X_te = X[:40], X[40:]

    lr  = LogisticRegression(max_iter=200, random_state=0).fit(X_bg, y[:40])
    rf  = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_bg, y[:40])

    class FakeLR:
        _clf = lr

    class FakeRF:
        _clf = rf

    shap_dict = compute_all_shap(
        {"ridge": FakeLR(), "random_forest": FakeRF()},
        X_bg, X_te, FEAT_NAMES,
        n_bg=20, n_test=20, n_bg_deep=10, nsamples_kernel=30,
    )

    assert "ridge" in shap_dict, "ridge must be in results"
    assert "random_forest" in shap_dict
    for name, sv in shap_dict.items():
        assert sv.shape[1] == N_FEAT, f"{name}: bad shape {sv.shape}"
        assert np.isfinite(sv).all(), f"{name}: non-finite SHAP values"

    # Charts
    plot_shap_heatmap(shap_dict, FEAT_NAMES, tmp_path, fmt="png")
    assert (tmp_path / "shap_heatmap.png").exists()

    plot_shap_summary_bars(shap_dict, FEAT_NAMES, tmp_path, fmt="png", top_n=5)
    assert (tmp_path / "shap_summary_bars.png").exists()

    plot_shap_group_comparison(shap_dict, FEAT_NAMES, tmp_path, fmt="png")
    assert (tmp_path / "shap_group_comparison.png").exists()

    save_shap_csv(shap_dict, FEAT_NAMES, tmp_path)
    csv_path = tmp_path / "shap_importance.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path, index_col=0)
    assert set(["ridge", "random_forest"]).issubset(df.columns)
    assert "group" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  CLI help smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_compute_shap_help():
    from click.testing import CliRunner
    from src.cli.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["compute-shap", "--help"])
    assert result.exit_code == 0, result.output
    assert "SHAP" in result.output or "shap" in result.output.lower()
