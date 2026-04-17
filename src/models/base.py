"""
src/models/base.py
──────────────────
Abstract base class for all stock-picking classifiers.

Every model in this project must:
  1. Implement ``fit(X_train, y_train, X_val, y_val)``.
  2. Implement ``predict_proba(X)`` returning shape (n_samples, 2) where
     column 1 is the outperformance probability.
  3. Expose a ``name`` property.
  4. Support optional ``feature_importances_`` / ``coef_`` attributes for
     explainability.

The interface is intentionally minimal so that sklearn, XGBoost, and PyTorch
models can all sit behind the same API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseStockModel(ABC):
    """Abstract base class for all stock-picking models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in reports and logs."""
        ...

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> "BaseStockModel":
        """Train the model.

        Parameters
        ----------
        X_train, y_train:
            Training features and labels (0/1).
        X_val, y_val:
            Validation features/labels (used for early stopping / CV).

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def predict_proba(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Return (n_samples, 2) array of class probabilities.

        Column 0 = P(underperform), Column 1 = P(outperform).
        """
        ...

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Hard classification at threshold 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # Optional: subclasses should override if available
    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return None

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


def get_model(name: str, cfg) -> "BaseStockModel":  # type: ignore[return]
    """Factory function: return the correct model instance for *name*.

    Parameters
    ----------
    name:
        One of the keys in ``cfg.models.enabled``.
    cfg:
        Project Config object.
    """
    from src.models.linear_models import (
        ElasticNetModel,
        LassoModel,
        PCALogisticModel,
        RidgeModel,
    )
    from src.models.tree_models import RandomForestModel, XGBoostModel
    from src.models.neural_models import DNNModel, LSTMModel

    models = {
        "ridge": lambda: RidgeModel(cfg),
        "lasso": lambda: LassoModel(cfg),
        "elasticnet": lambda: ElasticNetModel(cfg),
        "pca_logistic": lambda: PCALogisticModel(cfg),
        "random_forest": lambda: RandomForestModel(cfg),
        "xgboost": lambda: XGBoostModel(cfg),
        "dnn": lambda: DNNModel(cfg),
        "lstm": lambda: LSTMModel(cfg),
    }
    if name not in models:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(models.keys())}"
        )
    return models[name]()
