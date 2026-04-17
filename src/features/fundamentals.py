"""
src/features/fundamentals.py
────────────────────────────
Selects and renames fundamental columns from the panel built by
``src.data.loaders`` and optionally computes any cross-sectional
transformations needed.

This module is kept deliberately thin: all the heavy lifting (download,
lag, broadcast) already lives in ``src.data.fundamentals``.  Here we just
provide a clean list of the columns to use as model inputs and a helper to
log feature availability vs the paper.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Ordered list of fundamental feature columns that the paper expects.
# If a column is missing from the panel it is recorded in the availability
# report but does NOT cause an error.
FUNDAMENTAL_FEATURE_COLS: list[str] = [
    "market_cap",
    "book_to_market",
    "eps_growth",
    "leverage",
    "roe",              # proxy for ROIC
    "forward_eps",
    "net_income_mc",
    "sales_ev",
    "cfo_mc",
    "fcfe_mc",
    "fcf_ev",
    "dividend_yield",
    "operating_margin",
    "profit_margin",
    "revenue_growth",
]

# Human-readable description for reporting
FUNDAMENTAL_DESCRIPTIONS: dict[str, str] = {
    "market_cap": "Market capitalisation (size factor)",
    "book_to_market": "Book-to-market ratio (value factor) – 1/P/B from yfinance",
    "eps_growth": "EPS growth (quality/growth) – yfinance earningsGrowth",
    "leverage": "Financial leverage – yfinance debtToEquity",
    "roe": "Return on equity (ROIC proxy) – yfinance returnOnEquity",
    "forward_eps": "Forward EPS estimate – yfinance forwardEps",
    "net_income_mc": "Net income / market cap (profitability) – derived",
    "sales_ev": "Sales / enterprise value (profitability) – derived",
    "cfo_mc": "Operating cashflow / market cap – derived",
    "fcfe_mc": "Free cashflow / market cap (FCFE proxy) – derived",
    "fcf_ev": "Free cashflow / enterprise value – derived",
    "dividend_yield": "Dividend yield – yfinance dividendYield",
    "operating_margin": "Operating margin – yfinance operatingMargins",
    "profit_margin": "Profitability margin – yfinance profitMargins",
    "revenue_growth": "Sales growth – yfinance revenueGrowth",
}

# Paper variables NOT available in Phase-1 (Phase-2 upgrade targets)
MISSING_FROM_PHASE1: dict[str, str] = {
    "earnings_variability": "deviation from earnings trend – no time-series EPS in yfinance",
    "roic_exact": "exact ROIC – needs capital invested from balance sheet",
    "asset_growth": "asset growth – needs sequential balance sheet data",
    "cfi_ev": "cash from investing / EV – needs cash flow statement history",
    "employee_growth": "employee growth – requires periodic headcount data",
}


def get_fundamental_feature_cols(panel: pd.DataFrame) -> list[str]:
    """Return fundamental column names that are *actually present* in *panel*."""
    available = [c for c in FUNDAMENTAL_FEATURE_COLS if c in panel.columns]
    missing = [c for c in FUNDAMENTAL_FEATURE_COLS if c not in panel.columns]
    if missing:
        logger.debug("Fundamental features absent from panel: %s", missing)
    return available


def feature_availability_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a feature availability table comparing paper vs implementation.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, category, status, notes, coverage_pct]
    """
    rows = []
    for col, desc in FUNDAMENTAL_DESCRIPTIONS.items():
        if col in panel.columns:
            cov = panel[col].notna().mean() if not panel.empty else float("nan")
            rows.append(
                {
                    "feature": col,
                    "category": "fundamental",
                    "status": "available",
                    "notes": desc,
                    "coverage_pct": round(cov * 100, 1),
                }
            )
        else:
            rows.append(
                {
                    "feature": col,
                    "category": "fundamental",
                    "status": "missing",
                    "notes": desc,
                    "coverage_pct": 0.0,
                }
            )

    for col, reason in MISSING_FROM_PHASE1.items():
        rows.append(
            {
                "feature": col,
                "category": "fundamental",
                "status": "not_implemented_phase1",
                "notes": reason,
                "coverage_pct": 0.0,
            }
        )

    technical_cols = [
        "mom_12m", "mom_6m", "mom_1m",
        "rel_mom_12m", "rel_mom_6m", "rel_mom_1m",
        "log_price_ma200", "log_price_ma100", "log_price_ma50",
        "beta_12m",
        "vol_12m", "vol_6m", "vol_1m",
        "rsi_14", "rsi_9", "rsi_3",
        "log_price_bb_upper", "log_price_bb_lower",
        "ret_lag1", "ret_lag2",
        "usd_volume",
    ]
    for col in technical_cols:
        if col in panel.columns:
            cov = panel[col].notna().mean() if not panel.empty else float("nan")
            rows.append(
                {
                    "feature": col,
                    "category": "technical",
                    "status": "available",
                    "notes": "computed from yfinance weekly OHLCV",
                    "coverage_pct": round(cov * 100, 1),
                }
            )
        else:
            rows.append(
                {
                    "feature": col,
                    "category": "technical",
                    "status": "missing",
                    "notes": "check feature pipeline",
                    "coverage_pct": 0.0,
                }
            )

    return pd.DataFrame(rows).sort_values(["category", "status"])
