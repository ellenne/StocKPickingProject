"""
src/models/tree_models.py
─────────────────────────
Implements:
  - RandomForestModel : sklearn RandomForestClassifier with grid-search CV
  - XGBoostModel      : XGBoost gradient boosting classifier

Hyperparameter grids follow Appendix Table 1 of the paper.
Grid search uses TimeSeriesSplit (no random shuffling of time-ordered data).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.models.base import BaseStockModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseStockModel):
    """Random forest classifier with time-series-aware hyperparameter search."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        mc = cfg._raw["models"]["random_forest"]
        self._param_grid = {
            "n_estimators": mc.get("n_estimators_grid", [100, 250, 500]),
            "max_depth": mc.get("max_depth_grid", [3, 5, 7, 10]),
            "min_samples_leaf": mc.get("min_samples_leaf_grid", [1, 3, 5]),
        }
        self._cv_folds = mc.get("cv_folds", 5)
        self._n_jobs = mc.get("n_jobs", -1)
        self._clf: RandomForestClassifier | None = None

    @property
    def name(self) -> str:
        return "random_forest"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("[%s] grid-searching hyperparameters …", self.name)
        base = RandomForestClassifier(
            random_state=self.cfg.seed,
            n_jobs=self._n_jobs,
            class_weight="balanced",
        )
        tscv = TimeSeriesSplit(n_splits=self._cv_folds)
        gs = GridSearchCV(
            base,
            self._param_grid,
            cv=tscv,
            scoring="neg_log_loss",
            n_jobs=self._n_jobs,
            refit=True,
            verbose=0,
        )
        gs.fit(np.asarray(X_train), np.asarray(y_train))
        self._clf = gs.best_estimator_
        logger.info("[%s] best params: %s", self.name, gs.best_params_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(np.asarray(X))

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return self._clf.feature_importances_ if self._clf else None


class XGBoostModel(BaseStockModel):
    """XGBoost gradient boosting classifier.

    The paper uses AdaBoost with 1000 iterations but notes XGBoost as
    equivalent in spirit.  We use XGBoost because it:
      - supports early stopping (prevents overfitting),
      - is faster on large datasets,
      - has the same hyperparameter grid as described in the appendix.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        mc = cfg._raw["models"]["xgboost"]
        self._param_grid = {
            "max_depth": mc.get("max_depth_grid", [3, 5, 7]),
            "min_child_weight": mc.get("min_child_weight_grid", [1, 3, 5]),
            "learning_rate": mc.get("eta_grid", [0.05, 0.1]),
            "colsample_bytree": mc.get("colsample_bytree_grid", [0.7, 1.0]),
            "gamma": mc.get("gamma_grid", [0, 0.01]),
        }
        self._n_estimators = mc.get("n_estimators", 1000)
        self._early_stopping = mc.get("early_stopping_rounds", 50)
        self._subsample = mc.get("subsample", 0.5)
        self._cv_folds = mc.get("cv_folds", 5)
        self._clf = None

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        logger.info("[%s] grid-searching hyperparameters …", self.name)
        X_np = np.asarray(X_train)
        y_np = np.asarray(y_train)

        n_candidates = np.prod([len(v) for v in self._param_grid.values()])
        logger.info("[%s] grid size = %d parameter combinations", self.name, n_candidates)

        # Use a single best-effort fit with default params if grid is tiny (1 combo)
        if n_candidates == 1:
            params = {k: v[0] for k, v in self._param_grid.items()}
            self._clf = XGBClassifier(
                **params,
                n_estimators=self._n_estimators,
                subsample=self._subsample,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=self.cfg.seed,
                n_jobs=-1,
                verbosity=0,
            )
            if X_val is not None and y_val is not None:
                self._clf.set_params(early_stopping_rounds=self._early_stopping)
                self._clf.fit(
                    X_np, y_np,
                    eval_set=[(np.asarray(X_val), np.asarray(y_val))],
                    verbose=False,
                )
            else:
                self._clf.fit(X_np, y_np)
            return self

        # Use sklearn GridSearchCV with TimeSeriesSplit (no data leakage)
        tscv = TimeSeriesSplit(n_splits=self._cv_folds)
        base = XGBClassifier(
            n_estimators=self._n_estimators,
            subsample=self._subsample,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=self.cfg.seed,
            n_jobs=-1,
            verbosity=0,
        )
        gs = GridSearchCV(
            base,
            self._param_grid,
            cv=tscv,
            scoring="neg_log_loss",
            n_jobs=1,   # XGBoost already uses n_jobs=-1 internally
            refit=True,
            verbose=0,
        )
        gs.fit(X_np, y_np)
        self._clf = gs.best_estimator_
        logger.info("[%s] best params: %s", self.name, gs.best_params_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(np.asarray(X))

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return self._clf.feature_importances_ if self._clf else None
