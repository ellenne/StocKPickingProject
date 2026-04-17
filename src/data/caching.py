"""
src/data/caching.py
───────────────────
Lightweight parquet-based cache helpers.

Every raw download is persisted as a .parquet file keyed by a stable name.
Re-runs skip the network call unless *force_refresh* is True.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def cached_download(
    name: str,
    fetch_fn: Callable[[], pd.DataFrame],
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a cached DataFrame or call *fetch_fn* and cache the result.

    Parameters
    ----------
    name:
        Stable cache key (no special characters, used as filename).
    fetch_fn:
        Zero-argument callable that fetches and returns a ``DataFrame``.
    cache_dir:
        Directory where parquet files are stored.
    force_refresh:
        If ``True``, always call *fetch_fn* even if a cache file exists.

    Returns
    -------
    pd.DataFrame
    """
    path = _cache_path(cache_dir, name)
    if path.exists() and not force_refresh:
        logger.info("Cache hit  → %s", path.name)
        return pd.read_parquet(path)

    logger.info("Cache miss → fetching '%s'", name)
    df = fetch_fn()
    if df is not None and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("Saved to cache: %s (rows=%d)", path.name, len(df))
    else:
        logger.warning("fetch_fn returned empty result for '%s'; not cached.", name)
    return df


def invalidate(name: str, cache_dir: Path) -> None:
    """Delete a cached file so the next call re-fetches."""
    path = _cache_path(cache_dir, name)
    if path.exists():
        path.unlink()
        logger.info("Cache invalidated: %s", path.name)
