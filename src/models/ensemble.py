"""
src/models/ensemble.py
──────────────────────
Ensemble model that averages outperformance probabilities across all
individual models (simple mean combination).

The paper (Section 3.7) uses the simple average of individual model
probabilities and shows it consistently outperforms any single model
(Ensemble Sharpe = 0.84 for N=50, the best of all models).

The ``EnsembleModel`` wraps a list of already-fitted ``BaseStockModel``
instances and averages their ``predict_proba`` outputs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base import BaseStockModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseStockModel):
    """Simple average ensemble over a list of trained models."""

    def __init__(self, models: list[BaseStockModel]) -> None:
        if not models:
            raise ValueError("EnsembleModel requires at least one sub-model.")
        self._models = models
        logger.info(
            "Ensemble created with %d models: %s",
            len(models),
            [m.name for m in models],
        )

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def sub_models(self) -> list[BaseStockModel]:
        return self._models

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Ensemble does not re-train; sub-models should already be fitted."""
        logger.info(
            "[ensemble] using %d pre-trained sub-models", len(self._models)
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return mean probability across sub-models."""
        probas = np.stack(
            [m.predict_proba(X) for m in self._models], axis=0
        )  # shape (n_models, n_samples, 2)
        return probas.mean(axis=0)  # shape (n_samples, 2)

    def individual_probas(
        self, X
    ) -> dict[str, np.ndarray]:
        """Return per-model probability arrays (useful for diagnostics)."""
        return {m.name: m.predict_proba(X)[:, 1] for m in self._models}
