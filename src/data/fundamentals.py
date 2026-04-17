"""
src/data/fundamentals.py
────────────────────────
Downloads and cleans fundamental / financial-statement data.

Data source strategy (Phase 1)
───────────────────────────────
yfinance ``Ticker.info`` provides a single-snapshot dict with ~150 fields
covering most of the paper's fundamental variables.  Fields are fetched once
per ticker, cached, and then aligned to the weekly panel.

Because ``yfinance.info`` returns *current* fundamentals (not time-series),
Phase-1 treats fundamentals as **static** inputs that are repeated across all
weeks.  The 3-month lookahead lag is applied by shifting the entire block
forward by ``fundamentals_lag_months`` months relative to the price date.

Phase-2 upgrade path: replace ``_fetch_fundamentals_yfinance`` with a
time-series source (e.g. SimFin, SEC EDGAR XBRL, or FMP) to get quarterly
snapshots and eliminate the static approximation.

Available fundamental fields mapped to paper variables
───────────────────────────────────────────────────────
Paper variable               yfinance.info key (best match)
─────────────────────────── ─────────────────────────────────────────
Market cap                   marketCap
Book-to-market               (1/priceToBook)
EPS growth                   earningsGrowth
Earnings variability         (not available → dropped)
Leverage                     debtToEquity
ROIC                         returnOnEquity  (proxy)
Forward EPS estimate         forwardEps
Net income / market cap      (netIncomeToCommon / marketCap)
Sales / EV                   (totalRevenue / enterpriseValue)
CFO / market cap             (operatingCashflow / marketCap)
FCFE / market cap            (freeCashflow / marketCap)
FCF / EV                     (freeCashflow / enterpriseValue)
Dividend yield               dividendYield
Operating margin             operatingMargins
Profitability margin         profitMargins
Asset growth                 (not direct → dropped / proxied)
Cash from investing / EV     (not direct → dropped)
Employee growth              (not direct → dropped)
Sales growth                 revenueGrowth
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import pandas as pd
import yfinance as yf

from src.data.caching import cached_download

logger = logging.getLogger(__name__)

# Maps column names in our DataFrame to yfinance.info keys
_INFO_FIELD_MAP: dict[str, str] = {
    "market_cap": "marketCap",
    "book_to_market": None,          # derived: 1 / priceToBook
    "eps_growth": "earningsGrowth",
    "leverage": "debtToEquity",
    "roe": "returnOnEquity",         # proxy for ROIC
    "forward_eps": "forwardEps",
    "net_income_mc": None,           # derived: netIncomeToCommon / marketCap
    "sales_ev": None,                # derived: totalRevenue / enterpriseValue
    "cfo_mc": None,                  # derived: operatingCashflow / marketCap
    "fcfe_mc": None,                 # derived: freeCashflow / marketCap
    "fcf_ev": None,                  # derived: freeCashflow / enterpriseValue
    "dividend_yield": "dividendYield",
    "operating_margin": "operatingMargins",
    "profit_margin": "profitMargins",
    "revenue_growth": "revenueGrowth",
    # raw fields needed for derived computations
    "_price_to_book": "priceToBook",
    "_net_income": "netIncomeToCommon",
    "_total_revenue": "totalRevenue",
    "_enterprise_value": "enterpriseValue",
    "_op_cashflow": "operatingCashflow",
    "_free_cashflow": "freeCashflow",
}


def _fetch_info_single(ticker: str, max_retries: int = 3) -> dict:
    """Fetch yfinance.info dict for one ticker with retries."""
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if info and len(info) > 5:
                return info
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance info error %s (attempt %d): %s", ticker, attempt + 1, exc)
        time.sleep(1.5 ** attempt)
    return {}


def _info_to_row(ticker: str, info: dict) -> dict:
    """Extract the fields we need from a yfinance info dict."""
    row: dict = {"ticker": ticker}
    for col, yf_key in _INFO_FIELD_MAP.items():
        if yf_key is not None:
            row[col] = info.get(yf_key)
        else:
            row[col] = None  # will be filled in derived step

    # ── derived fields ────────────────────────────────────────────────────
    mc = info.get("marketCap") or None
    ev = info.get("enterpriseValue") or None
    ptb = info.get("priceToBook") or None
    ni = info.get("netIncomeToCommon") or None
    rev = info.get("totalRevenue") or None
    ocf = info.get("operatingCashflow") or None
    fcf = info.get("freeCashflow") or None

    row["book_to_market"] = (1.0 / ptb) if ptb and ptb != 0 else None
    row["net_income_mc"] = (ni / mc) if ni is not None and mc else None
    row["sales_ev"] = (rev / ev) if rev is not None and ev else None
    row["cfo_mc"] = (ocf / mc) if ocf is not None and mc else None
    row["fcfe_mc"] = (fcf / mc) if fcf is not None and mc else None
    row["fcf_ev"] = (fcf / ev) if fcf is not None and ev else None

    # Drop internal raw fields
    for k in list(row):
        if k.startswith("_"):
            del row[k]

    return row


def download_fundamentals(
    tickers: Sequence[str],
    cache_dir,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download fundamentals for all tickers.

    Returns
    -------
    pd.DataFrame
        One row per ticker, fundamental columns as float.
    """
    def _fetch_all():
        rows = []
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info("Fetching fundamentals … %d/%d", i, len(tickers))
            info = _fetch_info_single(ticker)
            rows.append(_info_to_row(ticker, info))
            time.sleep(0.1)  # polite rate limit
        return pd.DataFrame(rows).set_index("ticker")

    df = cached_download(
        f"fundamentals_{len(list(tickers))}",
        _fetch_all,
        cache_dir,
        force_refresh,
    )
    return df


def align_fundamentals_to_panel(
    fund_df: pd.DataFrame,
    weekly_dates: pd.DatetimeIndex,
    lag_months: int = 3,
) -> pd.DataFrame:
    """Broadcast static fundamentals across the weekly date panel.

    Because Phase-1 fundamentals are snapshots (not time-series), each ticker
    gets the same row of fundamentals for all weeks.  The 3-month lag means
    fundamentals are first available at date ``report_date + lag``.  Since we
    don't have quarterly report dates in Phase-1, we simply apply the lag by
    shifting the entire panel backward by *lag_months* months – i.e. the first
    ``lag_months`` months of the panel have missing fundamental data.

    Parameters
    ----------
    fund_df:
        (n_tickers × n_features) DataFrame with ticker as index.
    weekly_dates:
        DatetimeIndex of the weekly price panel.
    lag_months:
        Lookahead buffer in months.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker) with fundamental columns.
    """
    tickers = fund_df.index.tolist()
    rows = []
    for date in weekly_dates:
        for ticker in tickers:
            if ticker in fund_df.index:
                row = fund_df.loc[ticker].to_dict()
                row["date"] = date
                row["ticker"] = ticker
                rows.append(row)

    panel = pd.DataFrame(rows).set_index(["date", "ticker"])

    # Apply the lag: zero out fundamentals for the first lag_months
    cutoff = weekly_dates[0] + pd.DateOffset(months=lag_months)
    mask = panel.index.get_level_values("date") < cutoff
    panel.loc[mask] = float("nan")

    logger.info(
        "Fundamental panel shape: %s  (lag=%d months)",
        panel.shape,
        lag_months,
    )
    return panel
