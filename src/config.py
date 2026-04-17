"""
src/config.py
─────────────
Central configuration loader.  Reads configs/default.yaml (and an optional
user-supplied override file) and exposes a single ``Config`` dataclass that
is consumed throughout the project.

Usage
-----
    from src.config import load_config
    cfg = load_config()           # uses configs/default.yaml
    cfg = load_config("my.yaml")  # merges my.yaml on top of defaults
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Resolve project root as the parent of this file's parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (base is mutated in place)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def load_config(override_path: str | Path | None = None) -> "Config":
    """Load and validate configuration.

    Parameters
    ----------
    override_path:
        Optional path to a YAML file whose values override the defaults.

    Returns
    -------
    Config
        Validated configuration object.
    """
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_path, "r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if override_path is not None:
        with open(override_path, "r") as fh:
            overrides: dict[str, Any] = yaml.safe_load(fh)
        raw = _deep_merge(raw, overrides)
        logger.info("Config overrides applied from %s", override_path)

    return Config(raw)


class Config:
    """Thin wrapper around the raw YAML dict providing attribute access."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        # Cache resolved dirs so callers get absolute Path objects
        self._resolve_paths()

    # ── attribute access ──────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        try:
            return self._raw[name]
        except KeyError:
            raise AttributeError(f"Config has no section '{name}'")

    def get(self, *keys: str, default: Any = None) -> Any:
        """Drill into nested keys, e.g. cfg.get('models','ridge','C_grid')."""
        node = self._raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ── path helpers ──────────────────────────────────────────────────────
    def _resolve_paths(self) -> None:
        d = self._raw["data"]
        self.cache_dir: Path = PROJECT_ROOT / d["cache_dir"]
        self.processed_dir: Path = PROJECT_ROOT / d["processed_dir"]
        self.outputs_dir: Path = PROJECT_ROOT / self._raw["outputs"]["dir"]
        for p in (self.cache_dir, self.processed_dir, self.outputs_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ── convenience properties ────────────────────────────────────────────
    @property
    def enabled_models(self) -> list[str]:
        return self._raw["models"]["enabled"]

    @property
    def top_n(self) -> int:
        return int(self._raw["portfolio"]["top_n"])

    @property
    def train_years(self) -> int:
        return int(self._raw["rolling"]["train_years"])

    @property
    def test_years(self) -> int:
        return int(self._raw["rolling"]["test_years"])

    @property
    def seed(self) -> int:
        return int(self._raw["rolling"]["seed"])

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(start={self._raw['data']['start_date']}, models={self.enabled_models})"
