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
from typing import Dict, List, Optional

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


class PerformanceWeightedEnsemble:
    """Performance-weighted ensemble based on rolling Sharpe ratios."""

    def __init__(self, lookback_windows: int = 3):
        """
        Args:
            lookback_windows: Number of past windows to use for weighting.
        """
        self.lookback_windows = lookback_windows
        self.performance_history: Dict[str, list] = {}
        self.weights: Dict[str, float] = {}

    def update_performance(self, model_name: str, returns: np.ndarray) -> None:
        """Update performance history for a model."""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) * 52 / (np.std(returns) * np.sqrt(52))
        else:
            sharpe = 0.0

        self.performance_history[model_name].append(sharpe)
        if len(self.performance_history[model_name]) > self.lookback_windows:
            self.performance_history[model_name] = (
                self.performance_history[model_name][-self.lookback_windows:]
            )

    def calculate_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calculate performance-weighted ensemble weights."""
        avg_sharpes = {
            model: (
                float(np.mean(self.performance_history[model]))
                if model in self.performance_history and self.performance_history[model]
                else 0.0
            )
            for model in model_names
        }

        min_sharpe = min(avg_sharpes.values())
        adjusted = {k: v - min_sharpe + 0.1 for k, v in avg_sharpes.items()}
        total = sum(adjusted.values())
        if total > 0:
            return {k: v / total for k, v in adjusted.items()}
        return {k: 1.0 / len(model_names) for k in model_names}

    def predict_proba(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        model_names = list(model_predictions.keys())
        weights = self.calculate_weights(model_names)
        ensemble_pred = np.zeros_like(next(iter(model_predictions.values())))
        for model, pred in model_predictions.items():
            ensemble_pred += weights.get(model, 0) * pred
        return ensemble_pred
