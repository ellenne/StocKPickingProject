"""
src/reports/shap_analysis.py
─────────────────────────────
SHAP-based feature importance analysis reproducing the paper's Figure 8.

Explainer strategy per model type
──────────────────────────────────
  Ridge / Lasso / ElasticNet  → shap.LinearExplainer   (fast, exact)
  RandomForest / XGBoost      → shap.TreeExplainer     (fast, exact)
  DNN                         → shap.DeepExplainer     (fast, PyTorch)
  LSTM / PCA-Logistic / other → shap.KernelExplainer   (slow, model-agnostic;
                                                         wraps predict_proba[:, 1])

For LSTM, KernelExplainer is applied to a wrapper that treats the most-recent
week's feature values as the query (zero-padded context), giving interpretable
"last-week feature importance" in the same 2-D space as all other models.

All results are normalised to mean |SHAP| per feature per model so values are
comparable in the Figure-8 heatmap despite different output magnitudes.

Public API
──────────
  compute_all_shap(models, X_bg, X_test, feature_names, ...) → dict
  plot_shap_heatmap(shap_dict, feature_names, outputs_dir, fmt)
  plot_shap_summary_bars(shap_dict, feature_names, outputs_dir, fmt)
  plot_shap_beeswarm(shap_values, X_test, feature_names, model_name,
                     outputs_dir, fmt)
  save_shap_csv(shap_dict, feature_names, outputs_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Feature-group classification
# ─────────────────────────────────────────────────────────────────────────────

_TECH_PREFIXES = (
    "mom_", "rsi_", "vol_", "ma_", "bb_", "ret_", "beta_", "close_",
    "high_", "low_", "open_", "volume_", "atr_", "obv_", "cmf_",
    "stoch_", "macd_",
)
_SECTOR_PREFIX = "sector_"


def feature_group(name: str) -> str:
    """Return 'Technical', 'Fundamental', or 'Sector' for a feature name."""
    lo = name.lower()
    if lo.startswith(_SECTOR_PREFIX):
        return "Sector"
    if any(lo.startswith(p) for p in _TECH_PREFIXES):
        return "Technical"
    return "Fundamental"


def feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """Return dict grouping feature names by category."""
    groups: dict[str, list[str]] = {"Fundamental": [], "Technical": [], "Sector": []}
    for f in feature_names:
        groups[feature_group(f)].append(f)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
#  Per-model SHAP computation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_sv(sv) -> np.ndarray:
    """Accept SHAP output in any format and return (n_samples, n_features) float64.

    Handles:
      - list of arrays (older multi-class API): pick index [1] (outperform class)
      - shap.Explanation objects: extract .values
      - 3-D array (n_samples, n_features, n_classes): take class index [1]
      - 2-D array: returned as-is
    """
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    if hasattr(sv, "values"):
        sv = sv.values
    sv = np.asarray(sv, dtype=np.float64)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) – e.g. sklearn RF TreeExplainer
        sv = sv[:, :, 1]
    return sv


def _shap_linear(model, X_bg: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """LinearExplainer for Ridge / Lasso / ElasticNet."""
    import shap
    clf = getattr(model, "_clf", model)
    explainer = shap.LinearExplainer(clf, X_bg)
    return _normalise_sv(explainer.shap_values(X_test))


def _shap_tree(model, X_test: np.ndarray) -> np.ndarray:
    """TreeExplainer for RandomForest and XGBoost."""
    import shap
    clf = getattr(model, "_clf", model)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_test, check_additivity=False)
    return _normalise_sv(sv)


def _shap_deep_dnn(model, X_bg: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """DeepExplainer for DNNModel (PyTorch).

    DeepExplainer operates directly on the underlying _DNNNet module, which
    outputs raw logits.  SHAP values for logit[1] are a consistent proxy for
    feature importance for the outperform class.
    """
    import shap
    import torch

    net = model._net
    device = model._device
    net.eval()

    bg_t = torch.from_numpy(X_bg.astype(np.float32)).to(device)
    explainer = shap.DeepExplainer(net, bg_t)

    tst_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    sv = explainer.shap_values(tst_t)
    # sv is list of 2 arrays (n_test, n_features); take class-1
    return _normalise_sv(sv)


def _shap_kernel(
    predict_fn,
    X_bg: np.ndarray,
    X_test: np.ndarray,
    n_bg_samples: int,
    n_test_samples: int,
    nsamples_kernel: int = 100,
) -> np.ndarray:
    """KernelExplainer – model-agnostic, used for LSTM and PCA-Logistic.

    `predict_fn(X) -> float array of shape (n,)` must return P(outperform).
    """
    import shap

    bg_sample = shap.sample(X_bg, min(n_bg_samples, len(X_bg)))
    tst_sample = X_test[:n_test_samples]

    explainer = shap.KernelExplainer(predict_fn, bg_sample)
    sv = explainer.shap_values(tst_sample, nsamples=nsamples_kernel, silent=True)
    return _normalise_sv(sv)


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_shap(
    models: dict,
    X_bg: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    n_bg: int = 200,
    n_test: int = 500,
    n_bg_deep: int = 100,
    nsamples_kernel: int = 150,
) -> dict[str, np.ndarray]:
    """Compute SHAP values for every model in *models*.

    Parameters
    ----------
    models:
        ``{model_name: fitted BaseStockModel}`` – should NOT include ensemble.
    X_bg:
        Background / training data (preprocessed, 2-D float32 numpy array).
    X_test:
        Test data on which SHAP values are evaluated.
    feature_names:
        Column names aligned with axis-1 of X_bg / X_test.
    n_bg:
        Number of background samples (sub-sampled from X_bg) used for
        LinearExplainer and as context for KernelExplainer.
    n_test:
        Maximum test rows evaluated by slow explainers.
    n_bg_deep:
        Background samples for DeepExplainer.
    nsamples_kernel:
        Number of coalition samples for KernelExplainer (higher = more
        accurate, slower).

    Returns
    -------
    dict mapping model_name → (n_eval, n_features) float64 SHAP array.
    """
    rng = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_bg),  min(n_bg, len(X_bg)),  replace=False)
    tst_idx = rng.choice(len(X_test), min(n_test, len(X_test)), replace=False)
    X_bg_s  = X_bg[bg_idx]
    X_tst_s = X_test[tst_idx]

    bg_deep_idx = rng.choice(len(X_bg), min(n_bg_deep, len(X_bg)), replace=False)
    X_bg_deep   = X_bg[bg_deep_idx]

    results: dict[str, np.ndarray] = {}

    for name, model in models.items():
        logger.info("[shap] computing for model: %s", name)
        try:
            if name in ("ridge", "lasso", "elasticnet"):
                sv = _shap_linear(model, X_bg_s, X_tst_s)

            elif name in ("random_forest", "xgboost"):
                sv = _shap_tree(model, X_tst_s)

            elif name == "dnn" and hasattr(model, "_net") and model._net is not None:
                sv = _shap_deep_dnn(model, X_bg_deep, X_tst_s)

            else:
                # LSTM, PCA-Logistic, or any other model
                def _pred(x: np.ndarray) -> np.ndarray:
                    return model.predict_proba(x)[:, 1]

                sv = _shap_kernel(
                    _pred, X_bg_s, X_tst_s, n_bg, n_test, nsamples_kernel
                )

            if sv.ndim == 1:
                sv = sv.reshape(1, -1)

            if sv.shape[1] != len(feature_names):
                logger.warning(
                    "[shap] %s: sv width %d != n_features %d – skipping.",
                    name, sv.shape[1], len(feature_names),
                )
                continue

            results[name] = sv
            logger.info("[shap] %s: done (%d samples)", name, len(sv))

        except Exception as exc:  # noqa: BLE001
            logger.error("[shap] %s failed: %s", name, exc, exc_info=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Charts
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = {
    "Fundamental": "#3a86ff",
    "Technical":   "#ff006e",
    "Sector":      "#8338ec",
}

_MODEL_ORDER = [
    "ridge", "lasso", "elasticnet", "pca_logistic",
    "random_forest", "xgboost", "dnn", "lstm",
]


def _mean_abs(sv: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature across the test sample."""
    return np.mean(np.abs(sv), axis=0)


def _save(fig: plt.Figure, path: Path, fmt: str = "png") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("SHAP chart saved: %s", path.with_suffix(f".{fmt}"))


def plot_shap_heatmap(
    shap_dict: dict[str, np.ndarray],
    feature_names: list[str],
    outputs_dir: Path,
    fmt: str = "png",
) -> None:
    """Figure-8 style heatmap: features × models, colour = mean |SHAP|.

    Features are ordered by group (Fundamental → Technical → Sector) and then
    by descending mean importance across all models.  Columns are normalised
    per model so relative within-model rankings are visible despite different
    absolute SHAP scales.
    """
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("[shap] seaborn not installed – skipping heatmap.")
        return

    if not shap_dict:
        logger.warning("[shap] no SHAP results – skipping heatmap.")
        return

    model_order = [m for m in _MODEL_ORDER if m in shap_dict]
    model_order += [m for m in shap_dict if m not in model_order]

    # Build importance matrix
    rows = {}
    for model_name in model_order:
        rows[model_name] = _mean_abs(shap_dict[model_name])
    df = pd.DataFrame(rows, index=feature_names)

    # Normalise each column to [0, 1] for visual comparability
    df = df.div(df.max().replace(0, 1))

    # Sort features: group first, then descending mean importance
    groups = feature_groups(feature_names)
    group_order = ["Fundamental", "Technical", "Sector"]
    ordered_feats: list[str] = []
    group_boundaries: list[int] = []
    for grp in group_order:
        feats = groups.get(grp, [])
        if not feats:
            continue
        feats_sorted = sorted(feats, key=lambda f: -df.loc[f].mean() if f in df.index else 0)
        ordered_feats.extend(feats_sorted)
        group_boundaries.append(len(ordered_feats))

    df = df.loc[[f for f in ordered_feats if f in df.index]]

    n_feats = len(df)
    fig_h = max(8, n_feats * 0.22)
    fig, ax = plt.subplots(figsize=(len(model_order) * 1.6 + 2, fig_h))

    sns.heatmap(
        df,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
        linecolor="white",
        cbar_kws={"label": "Normalised mean |SHAP|", "shrink": 0.7},
        yticklabels=True,
        xticklabels=[m.replace("_", " ").title() for m in df.columns],
    )

    # Group separator lines and right-margin labels
    prev = 0
    for grp, boundary in zip(group_order, group_boundaries):
        if not groups.get(grp):
            continue
        mid = (prev + boundary) / 2
        ax.axhline(boundary, color="black", linewidth=1.8)
        ax.text(
            len(model_order) + 0.15, mid,
            grp,
            va="center", ha="left", fontsize=9, fontweight="bold",
            color=_PALETTE.get(grp, "black"),
            transform=ax.get_yaxis_transform(),
        )
        prev = boundary

    ax.set_title(
        "Feature Importance (mean |SHAP|) – Figure 8 style",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    fig.tight_layout()
    _save(fig, outputs_dir / "shap_heatmap", fmt)


def plot_shap_summary_bars(
    shap_dict: dict[str, np.ndarray],
    feature_names: list[str],
    outputs_dir: Path,
    fmt: str = "png",
    top_n: int = 20,
) -> None:
    """Horizontal bar charts: top-N features per model by mean |SHAP|."""
    model_order = [m for m in _MODEL_ORDER if m in shap_dict]
    model_order += [m for m in shap_dict if m not in model_order]

    n_models = len(model_order)
    if n_models == 0:
        return

    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax, model_name in zip(flat_axes, model_order):
        sv    = shap_dict[model_name]
        imp   = _mean_abs(sv)
        order = np.argsort(imp)[::-1][:top_n]
        names = [feature_names[i] for i in reversed(order)]
        vals  = imp[list(reversed(order))]
        colors = [_PALETTE.get(feature_group(f), "#999") for f in names]
        ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(model_name.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |SHAP|", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="x", alpha=0.3)

    # Legend for feature groups
    patches = [
        mpatches.Patch(color=c, label=g) for g, c in _PALETTE.items()
    ]
    flat_axes[0].legend(handles=patches, fontsize=8, loc="lower right")

    for ax_extra in flat_axes[n_models:]:
        ax_extra.set_visible(False)

    fig.suptitle(f"Top-{top_n} Features by Mean |SHAP| – All Models",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, outputs_dir / "shap_summary_bars", fmt)


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    outputs_dir: Path,
    fmt: str = "png",
    top_n: int = 20,
) -> None:
    """SHAP beeswarm / dot plot for a single model (top-N features)."""
    try:
        import shap
    except ImportError:
        return

    imp   = _mean_abs(shap_values)
    order = np.argsort(imp)[::-1][:top_n]

    exp = shap.Explanation(
        values=shap_values[:, order],
        data=X_test[:len(shap_values), :][:, order],
        feature_names=[feature_names[i] for i in order],
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    shap.plots.beeswarm(exp, max_display=top_n, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_title(
        f"SHAP Summary – {model_name.replace('_', ' ').title()}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_path = outputs_dir / f"shap_beeswarm_{model_name}"
    _save(fig, out_path, fmt)


def plot_shap_group_comparison(
    shap_dict: dict[str, np.ndarray],
    feature_names: list[str],
    outputs_dir: Path,
    fmt: str = "png",
) -> None:
    """Grouped bar chart: total importance of Technical vs Fundamental vs Sector."""
    groups = feature_groups(feature_names)
    group_order = ["Fundamental", "Technical", "Sector"]

    model_order = [m for m in _MODEL_ORDER if m in shap_dict]
    model_order += [m for m in shap_dict if m not in model_order]

    # For each model, compute sum of mean |SHAP| per group
    data: dict[str, list[float]] = {g: [] for g in group_order}
    for model_name in model_order:
        sv  = shap_dict[model_name]
        imp = _mean_abs(sv)
        total = imp.sum() or 1.0
        for grp in group_order:
            idxs = [i for i, f in enumerate(feature_names) if f in groups[grp]]
            data[grp].append(imp[idxs].sum() / total if idxs else 0.0)

    x  = np.arange(len(model_order))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(model_order) * 1.5), 5))

    for i, grp in enumerate(group_order):
        ax.bar(
            x + (i - 1) * w,
            data[grp],
            width=w,
            label=grp,
            color=_PALETTE[grp],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in model_order],
        rotation=25, ha="right", fontsize=9,
    )
    ax.set_ylabel("Fraction of total mean |SHAP|", fontsize=10)
    ax.set_title("Feature-Group Importance by Model", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, outputs_dir / "shap_group_comparison", fmt)


# ─────────────────────────────────────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_shap_csv(
    shap_dict: dict[str, np.ndarray],
    feature_names: list[str],
    outputs_dir: Path,
) -> None:
    """Save mean |SHAP| per (feature, model) to a CSV for further analysis."""
    rows = {}
    for model_name, sv in shap_dict.items():
        rows[model_name] = _mean_abs(sv)

    df = pd.DataFrame(rows, index=feature_names)
    df.index.name = "feature"
    df["group"] = [feature_group(f) for f in feature_names]
    path = outputs_dir / "shap_importance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("SHAP importance CSV saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
#  High-level entry point (called from CLI)
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    models: dict,
    X_bg: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    outputs_dir: Path,
    fmt: str = "png",
    n_bg: int = 200,
    n_test: int = 500,
    n_bg_deep: int = 100,
    nsamples_kernel: int = 150,
    top_n_bars: int = 20,
) -> dict[str, np.ndarray]:
    """Full SHAP pipeline: compute values, save all charts and CSV.

    Returns the raw SHAP dict for any downstream use.
    """
    shap_dir = outputs_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[shap] starting analysis for %d models", len(models))
    shap_dict = compute_all_shap(
        models, X_bg, X_test, feature_names,
        n_bg=n_bg, n_test=n_test,
        n_bg_deep=n_bg_deep, nsamples_kernel=nsamples_kernel,
    )

    if not shap_dict:
        logger.warning("[shap] no results to plot.")
        return shap_dict

    plot_shap_heatmap(shap_dict, feature_names, shap_dir, fmt)
    plot_shap_summary_bars(shap_dict, feature_names, shap_dir, fmt, top_n=top_n_bars)
    plot_shap_group_comparison(shap_dict, feature_names, shap_dir, fmt)
    save_shap_csv(shap_dict, feature_names, shap_dir)

    # Per-model beeswarm (only for models where we have raw SHAP values)
    for model_name, sv in shap_dict.items():
        X_tst_sample = X_test[:len(sv)]
        try:
            plot_shap_beeswarm(sv, X_tst_sample, feature_names, model_name,
                               shap_dir, fmt, top_n=top_n_bars)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[shap] beeswarm for %s failed: %s", model_name, exc)

    logger.info("[shap] analysis complete – outputs in %s", shap_dir)
    return shap_dict
