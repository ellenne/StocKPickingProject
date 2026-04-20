"""
src/data/prices.py
──────────────────
Downloads and caches weekly OHLCV price data for a list of tickers.

Key design decisions
────────────────────
* Uses yfinance with auto-adjusting for splits / dividends.
* Aligns to Wednesday closes per the paper; if Wednesday is a non-trading day
  the next available trading day within the same week is used.
* Returns are computed as log-returns of adjusted close.
* Missing tickers (delisted etc.) are silently skipped with a warning.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.caching import cached_download

logger = logging.getLogger(__name__)

# Day-of-week offset for Wednesday resample anchor
_WEDNESDAY = "W-WED"


def _fetch_prices_batch(
    tickers: list[str],
    start: str,
    end: str,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Download daily adjusted close + volume for *tickers* via yfinance.

    yfinance ≥ 1.3.0 always returns a MultiIndex with level name 'Ticker'
    regardless of whether one or many tickers are requested.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            return raw
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance download error (attempt %d): %s", attempt + 1, exc)
            attempt += 1
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"Failed to download prices after {max_retries} retries.")


def _daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (Wednesday) frequency.

    Wolff & Echterling (2022) Section 2: "If a Wednesday is a non-trading day
    (holiday), we use data of the next trading day available."

    Implementation
    --------------
    * Open / High / Low / Volume: period-based aggregates over the Thu–Wed window
      (first, max, min, sum respectively).
    * Close: the paper requires the NEXT available trading day when Wednesday is a
      holiday.  We achieve this with ``reindex(..., method="bfill")`` which maps
      each Wednesday to the first trading day >= that Wednesday within ±4 days
      (stays within the same week; never crosses into the following Wednesday).

    Note: the previous implementation used ``resample(..., Close="last")`` which
    picks the PREVIOUS trading day for Wednesday holidays – the opposite direction.
    """
    # ── Period aggregates for O / H / L / V ──────────────────────────────────
    agg_map = {k: v for k, v in
               {"Open": "first", "High": "max", "Low": "min", "Volume": "sum"}.items()
               if k in daily.columns}
    if agg_map:
        weekly = daily.resample(_WEDNESDAY).agg(agg_map)
    else:
        # No OHLV columns present – seed an empty frame with W-WED dates
        weekly = pd.DataFrame(index=daily.resample(_WEDNESDAY).first().index)

    # ── Close: next trading day on or after Wednesday (paper-faithful) ───────
    if "Close" in daily.columns:
        # bfill: for each Wednesday label, use the first available day >= that date
        # tolerance="4D": only look Thursday–Sunday (same week, never next Wednesday)
        close_next = daily[["Close"]].reindex(
            weekly.index, method="bfill", tolerance=pd.Timedelta("4D")
        )
        weekly["Close"] = close_next["Close"]

    weekly = weekly.dropna(how="all")
    return weekly


def download_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    cache_dir,
    force_refresh: bool = False,
    batch_size: int = 100,
) -> dict[str, pd.DataFrame]:
    """Download weekly OHLCV for each ticker.

    Parameters
    ----------
    tickers:
        Iterable of ticker strings.
    start, end:
        Date strings ``"YYYY-MM-DD"``.
    cache_dir:
        Path object where parquet files are stored.
    force_refresh:
        Ignore existing cache and re-download.
    batch_size:
        Download in chunks of this many tickers to avoid yfinance timeouts.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are tickers; each DataFrame has columns
        [Open, High, Low, Close, Volume] indexed by weekly date.
    """
    tickers = list(tickers)
    results: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    # Split into batches
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    for batch_idx, batch in enumerate(batches):
        logger.info(
            "Downloading prices: batch %d/%d (%d tickers)",
            batch_idx + 1,
            len(batches),
            len(batch),
        )
        cache_key = f"prices_batch_{batch_idx}_{start}_{end}"

        def _fetch(b=batch, s=start, e=end):
            return _fetch_prices_batch(b, s, e)

        raw = cached_download(cache_key, _fetch, cache_dir, force_refresh)
        if raw is None or raw.empty:
            missing.extend(batch)
            continue

        # yfinance always returns MultiIndex columns (Ticker, Price) or (Price, Ticker)
        # Use the level named 'Ticker' regardless of its position.
        if isinstance(raw.columns, pd.MultiIndex):
            ticker_level = raw.columns.names.index("Ticker") if "Ticker" in raw.columns.names else 1
            for ticker in batch:
                try:
                    tk_df = raw.xs(ticker, axis=1, level=ticker_level).copy()
                    tk_df = tk_df.dropna(how="all")
                    if tk_df.empty:
                        missing.append(ticker)
                        continue
                    weekly = _daily_to_weekly(tk_df)
                    results[ticker] = weekly
                except KeyError:
                    missing.append(ticker)
        else:
            # Flat columns (single ticker, older yfinance)
            tk = batch[0]
            weekly = _daily_to_weekly(raw.dropna(how="all"))
            if not weekly.empty:
                results[tk] = weekly
            else:
                missing.append(tk)

    if missing:
        logger.warning(
            "%d tickers had no downloadable data: %s",
            len(missing),
            missing[:20],
        )
    logger.info(
        "Price download complete: %d/%d tickers retrieved",
        len(results),
        len(tickers),
    )
    return results


def build_price_panel(
    price_dict: dict[str, pd.DataFrame],
    field: str = "Close",
) -> pd.DataFrame:
    """Stack per-ticker DataFrames into a (date × ticker) panel.

    Parameters
    ----------
    price_dict:
        Output of :func:`download_prices`.
    field:
        Which OHLCV column to extract (default ``"Close"``).

    Returns
    -------
    pd.DataFrame
        Shape (n_weeks, n_tickers).  Indexed by weekly date.
    """
    series = {}
    for ticker, df in price_dict.items():
        if field in df.columns:
            series[ticker] = df[field]
    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    logger.info(
        "Price panel: %d weeks × %d tickers", panel.shape[0], panel.shape[1]
    )
    return panel


def compute_weekly_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Log-returns from the weekly close panel.

    Returns
    -------
    pd.DataFrame
        Same shape as *price_panel*, first row is NaN.
    """
    return np.log(price_panel / price_panel.shift(1))
