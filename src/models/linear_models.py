"""
src/models/linear_models.py
───────────────────────────
Implements:
  - RidgeModel      : Ridge logistic regression  (L2 penalty)
  - LassoModel      : Lasso logistic regression  (L1 penalty)
  - ElasticNetModel : ElasticNet logistic regression
  - PCALogisticModel: PCA dimensionality reduction + logistic regression

All models use scikit-learn with 5-fold **time-series** cross-validation for
hyperparameter selection (no random shuffling of time-ordered data).

Hyperparameter grids follow Appendix Table 1 of the paper:
  Ridge / Lasso: 500 λ on log scale 1e-4 … 1e4
  (scikit-learn uses C = 1/λ, so we invert the grid)
  ElasticNet:   same λ grid × 20 α values in [0, 1]
  PCA:          n_components up to N explaining 90% variance

Paper note: sklearn's ``LogisticRegressionCV`` performs the λ search
internally and is more memory-efficient than a manual GridSearch.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.models.base import BaseStockModel

logger = logging.getLogger(__name__)

# C = 1 / λ; paper λ grid ∈ [1e-4, 1e4] → C ∈ [1e-4, 1e4] (symmetric)
_DEFAULT_C_GRID = np.logspace(-4, 4, 40).tolist()
_DEFAULT_CV = 5


class RidgeModel(BaseStockModel):
    """Ridge (L2) penalised logistic regression."""

    def __init__(self, cfg) -> None:  # type: ignore[annotation-unchecked]
        self.cfg = cfg
        model_cfg = cfg._raw["models"]["ridge"]
        c_grid = model_cfg.get("C_grid", _DEFAULT_C_GRID)
        cv_folds = model_cfg.get("cv_folds", _DEFAULT_CV)
        self._clf = LogisticRegressionCV(
            Cs=c_grid,
            cv=TimeSeriesSplit(n_splits=cv_folds),
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=cfg.seed,
        )

    @property
    def name(self) -> str:
        return "ridge"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("[%s] fitting …", self.name)
        self._clf.fit(X_train, y_train)
        logger.info("[%s] best C=%.4g", self.name, self._clf.C_[0])
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return self._clf.coef_


class LassoModel(BaseStockModel):
    """Lasso (L1) penalised logistic regression."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        model_cfg = cfg._raw["models"]["lasso"]
        c_grid = model_cfg.get("C_grid", _DEFAULT_C_GRID)
        cv_folds = model_cfg.get("cv_folds", _DEFAULT_CV)
        self._clf = LogisticRegressionCV(
            Cs=c_grid,
            cv=TimeSeriesSplit(n_splits=cv_folds),
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            scoring="neg_log_loss",
            random_state=cfg.seed,
        )

    @property
    def name(self) -> str:
        return "lasso"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("[%s] fitting …", self.name)
        self._clf.fit(X_train, y_train)
        logger.info("[%s] best C=%.4g", self.name, self._clf.C_[0])
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return self._clf.coef_


class ElasticNetModel(BaseStockModel):
    """ElasticNet logistic regression (L1 + L2 mix)."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        model_cfg = cfg._raw["models"]["elasticnet"]
        c_grid = model_cfg.get("C_grid", _DEFAULT_C_GRID)
        l1_ratios = model_cfg.get("l1_ratio_grid", [0.1, 0.5, 0.7, 0.9, 0.95, 1.0])
        cv_folds = model_cfg.get("cv_folds", _DEFAULT_CV)
        self._clf = LogisticRegressionCV(
            Cs=c_grid,
            cv=TimeSeriesSplit(n_splits=cv_folds),
            penalty="elasticnet",
            solver="saga",
            l1_ratios=l1_ratios,
            max_iter=2000,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=cfg.seed,
        )

    @property
    def name(self) -> str:
        return "elasticnet"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("[%s] fitting …", self.name)
        self._clf.fit(X_train, y_train)
        logger.info(
            "[%s] best C=%.4g l1_ratio=%.2f",
            self.name,
            self._clf.C_[0],
            self._clf.l1_ratio_[0],
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return self._clf.coef_


class PCALogisticModel(BaseStockModel):
    """PCA dimensionality reduction followed by logistic regression.

    The number of PCA components is selected by 5-fold cross-validation,
    starting from 1 up to N components that explain 90 % of variance (the
    elbow heuristic used in the paper).
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        model_cfg = cfg._raw["models"]["pca_logistic"]
        self._max_components = model_cfg.get("max_components", None)
        self._cv_folds = model_cfg.get("cv_folds", _DEFAULT_CV)
        self._pipeline: Pipeline | None = None
        self._best_n: int = 0

    @property
    def name(self) -> str:
        return "pca_logistic"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.model_selection import cross_val_score

        logger.info("[%s] selecting n_components …", self.name)
        X = np.asarray(X_train)

        # Elbow: find n explaining 90 % variance
        pca_full = PCA(n_components=None, random_state=self.cfg.seed)
        pca_full.fit(X)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        elbow = int(np.searchsorted(cumvar, 0.90)) + 1
        max_n = min(elbow, X.shape[1], self._max_components or elbow)

        best_score = -np.inf
        best_n = 1
        tscv = TimeSeriesSplit(n_splits=self._cv_folds)

        for n in range(1, max_n + 1):
            pipe = Pipeline(
                [
                    ("pca", PCA(n_components=n, random_state=self.cfg.seed)),
                    (
                        "clf",
                        LogisticRegressionCV(
                            Cs=_DEFAULT_C_GRID,
                            cv=tscv,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=500,
                            scoring="neg_log_loss",
                            n_jobs=-1,
                            random_state=self.cfg.seed,
                        ),
                    ),
                ]
            )
            scores = cross_val_score(
                pipe, X, y_train, cv=tscv, scoring="neg_log_loss", n_jobs=-1
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_n = n

        logger.info("[%s] best n_components=%d", self.name, best_n)
        self._best_n = best_n
        self._pipeline = Pipeline(
            [
                ("pca", PCA(n_components=best_n, random_state=self.cfg.seed)),
                (
                    "clf",
                    LogisticRegressionCV(
                        Cs=_DEFAULT_C_GRID,
                        cv=tscv,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=1000,
                        scoring="neg_log_loss",
                        n_jobs=-1,
                        random_state=self.cfg.seed,
                    ),
                ),
            ]
        )
        self._pipeline.fit(X, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._pipeline.predict_proba(np.asarray(X))
