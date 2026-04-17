"""
src/data/loaders.py
───────────────────
High-level orchestrator that coordinates price, volume, fundamental, and
sector data into a single aligned weekly panel stored on disk as parquet.

The resulting dataset has a MultiIndex (date, ticker) and contains:
  - OHLCV columns from prices
  - log_return (weekly)
  - all technical feature inputs (raw prices needed for indicator calc)
  - fundamental columns (lagged)
  - sector_* one-hot columns
  - target (label) column – computed separately in features/target.py

Calling ``build_dataset`` is idempotent: if the processed file already
exists and *force_refresh* is False it is loaded from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.caching import cached_download
from src.data.fundamentals import align_fundamentals_to_panel, download_fundamentals
from src.data.prices import build_price_panel, compute_weekly_returns, download_prices
from src.data.universe import Universe

logger = logging.getLogger(__name__)

_SECTOR_MAP_FIELD = "sector"      # yfinance.info key for sector
_SECTOR_DUMMIES_PREFIX = "sector_"


def _fetch_sector(tickers: list[str], cache_dir: Path, force_refresh: bool) -> pd.Series:
    """Fetch sector classification for each ticker."""
    import time
    import yfinance as yf

    def _fetch():
        rows = []
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info("Fetching sector … %d/%d", i, len(tickers))
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Unknown") or "Unknown"
            except Exception:  # noqa: BLE001
                sector = "Unknown"
            rows.append({"ticker": ticker, "sector": sector})
            time.sleep(0.05)
        return pd.DataFrame(rows).set_index("ticker")

    df = cached_download(f"sectors_{len(tickers)}", _fetch, cache_dir, force_refresh)
    return df["sector"] if "sector" in df.columns else pd.Series(dtype=str)


def build_dataset(
    cfg: Config,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download all data and assemble the weekly panel.

    Parameters
    ----------
    cfg:
        Loaded configuration object.
    force_refresh:
        Re-download even if cache files exist.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker), columns = price fields + fundamentals +
        sector dummies + log_return.
    """
    processed_path = cfg.processed_dir / "weekly_panel.parquet"
    if processed_path.exists() and not force_refresh:
        logger.info("Loading processed dataset from %s", processed_path)
        return pd.read_parquet(processed_path)

    universe = Universe(cfg)
    tickers = universe.all_tickers
    logger.info("Universe size: %d tickers", len(tickers))

    # ── 1. Prices ──────────────────────────────────────────────────────────
    data_cfg = cfg._raw["data"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")
    max_missing = data_cfg.get("max_missing_pct", 0.3)

    price_dict = download_prices(tickers, start, end, cfg.cache_dir, force_refresh)

    close_panel = build_price_panel(price_dict, "Close")
    open_panel = build_price_panel(price_dict, "Open")
    high_panel = build_price_panel(price_dict, "High")
    low_panel = build_price_panel(price_dict, "Low")
    vol_panel = build_price_panel(price_dict, "Volume")

    # Drop tickers with too many missing weeks
    n_weeks = len(close_panel)
    missing_frac = close_panel.isna().mean()
    bad_tickers = missing_frac[missing_frac > max_missing].index.tolist()
    if bad_tickers:
        logger.warning(
            "Dropping %d tickers with >%.0f%% missing price data",
            len(bad_tickers),
            max_missing * 100,
        )
    tickers_ok = [t for t in close_panel.columns if t not in bad_tickers]
    close_panel = close_panel[tickers_ok]
    open_panel = open_panel.reindex(columns=tickers_ok)
    high_panel = high_panel.reindex(columns=tickers_ok)
    low_panel = low_panel.reindex(columns=tickers_ok)
    vol_panel = vol_panel.reindex(columns=tickers_ok)

    weekly_dates = close_panel.index

    # ── 2. Log returns ─────────────────────────────────────────────────────
    ret_panel = compute_weekly_returns(close_panel)

    # ── 3. Fundamentals ─────────────────────────────────────────────────────
    # Download against the full original ticker list (uses the cached file),
    # then filter to only the tickers_ok subset.
    fund_raw_full = download_fundamentals(tickers, cfg.cache_dir, force_refresh)
    fund_raw = fund_raw_full.reindex(tickers_ok)
    lag_months = data_cfg.get("fundamentals_lag_months", 3)
    fund_panel = align_fundamentals_to_panel(fund_raw, weekly_dates, lag_months)

    # ── 4. Sectors ──────────────────────────────────────────────────────────
    sector_series = _fetch_sector(tickers_ok, cfg.cache_dir, force_refresh)

    # ── 5. Stack into long format ───────────────────────────────────────────
    logger.info("Stacking panels into long format …")
    dfs_to_concat = []
    for ticker in tickers_ok:
        if ticker not in close_panel.columns:
            continue
        tk_df = pd.DataFrame(
            {
                "close": close_panel[ticker],
                "open": open_panel[ticker] if ticker in open_panel.columns else float("nan"),
                "high": high_panel[ticker] if ticker in high_panel.columns else float("nan"),
                "low": low_panel[ticker] if ticker in low_panel.columns else float("nan"),
                "volume": vol_panel[ticker] if ticker in vol_panel.columns else float("nan"),
                "log_return": ret_panel[ticker] if ticker in ret_panel.columns else float("nan"),
            },
            index=weekly_dates,
        )
        tk_df["ticker"] = ticker
        tk_df.index.name = "date"
        dfs_to_concat.append(tk_df)

    panel = pd.concat(dfs_to_concat).reset_index()
    panel = panel.set_index(["date", "ticker"]).sort_index()

    # Merge fundamentals
    panel = panel.join(fund_panel, how="left")

    # Merge sectors → one-hot encode
    panel["sector_raw"] = panel.index.get_level_values("ticker").map(
        sector_series.to_dict()
    )
    sector_dummies = pd.get_dummies(panel["sector_raw"], prefix=_SECTOR_DUMMIES_PREFIX)
    panel = pd.concat([panel.drop(columns=["sector_raw"]), sector_dummies], axis=1)

    # ── 6. Persist ──────────────────────────────────────────────────────────
    panel.to_parquet(processed_path)
    logger.info(
        "Dataset saved → %s  (shape %s)",
        processed_path,
        panel.shape,
    )
    return panel
