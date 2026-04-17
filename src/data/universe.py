"""
src/data/universe.py
────────────────────
Builds the stock universe used for each rebalance date.

Phase-1 implementation
    Uses the *current* S&P 500 constituent list scraped from Wikipedia.
    This is **survivorship-biased** – the historical backtest will only see
    today's survivors.  The code is architected so a historical constituent
    CSV (date, ticker) can be dropped in via ``configs/default.yaml`` to
    replace this behaviour.

Historical constituent CSV format (when provided)
    date,ticker
    1999-01-06,AAPL
    1999-01-06,MSFT
    ...
    Each row means ticker was a member of the index on that date.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_WIKIPEDIA_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)


def get_sp500_tickers_wikipedia() -> list[str]:
    """Scrape the current S&P 500 constituents from Wikipedia.

    Returns
    -------
    list[str]
        Ticker symbols, dots converted to hyphens (yfinance convention).
    """
    import io
    import requests as req

    logger.info("Fetching current S&P 500 constituents from Wikipedia …")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = req.get(_WIKIPEDIA_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), header=0)
    df = tables[0]
    tickers: list[str] = (
        df["Symbol"].str.strip().str.replace(".", "-", regex=False).tolist()
    )
    logger.info("Found %d tickers", len(tickers))
    return sorted(set(tickers))


def load_historical_constituents(csv_path: str | Path) -> pd.DataFrame:
    """Load historical S&P 500 membership from a CSV.

    Parameters
    ----------
    csv_path:
        Path to CSV with columns ``[date, ticker]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (datetime), ``ticker`` (str).
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["ticker"] = df["ticker"].str.strip().str.replace(".", "-", regex=False)
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        "Loaded historical constituents: %d rows, %d unique tickers, "
        "date range %s – %s",
        len(df),
        df["ticker"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


class Universe:
    """Manages the stock universe over time.

    Parameters
    ----------
    cfg:
        Project config (``src.config.Config``).
    """

    def __init__(self, cfg) -> None:  # type: ignore[annotation-unchecked]
        self.cfg = cfg
        self._historical: pd.DataFrame | None = None
        self._current_tickers: list[str] | None = None
        self._load()

    def _load(self) -> None:
        universe_cfg = self.cfg._raw.get("universe", {})
        hist_csv = universe_cfg.get("historical_constituents_csv") if isinstance(universe_cfg, dict) else None

        if hist_csv:
            self._historical = load_historical_constituents(hist_csv)
            self._current_tickers = sorted(
                self._historical["ticker"].unique().tolist()
            )
        else:
            warn = universe_cfg.get("survivorship_bias_warning", True) if isinstance(universe_cfg, dict) else True
            if warn:
                logger.warning(
                    "⚠  No historical constituents CSV provided. "
                    "Using CURRENT S&P 500 members only – "
                    "backtest results will be SURVIVORSHIP-BIASED."
                )
            self._current_tickers = get_sp500_tickers_wikipedia()

    @property
    def all_tickers(self) -> list[str]:
        """All unique tickers in the universe."""
        return list(self._current_tickers or [])

    def tickers_at(self, date: pd.Timestamp) -> list[str]:
        """Return tickers that were members of the index on *date*.

        Falls back to the full current list when no historical data exists.
        """
        if self._historical is not None:
            subset = self._historical[self._historical["date"] <= date]
            if subset.empty:
                return []
            # Use the most recent snapshot before or on *date*
            last_date = subset["date"].max()
            return sorted(
                subset[subset["date"] == last_date]["ticker"].tolist()
            )
        return list(self._current_tickers or [])
