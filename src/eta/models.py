"""Model factories. Every model takes the feature DataFrame with raw categoricals, so the same
frame feeds the baseline, the linear model, gradient boosting and the quantile models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from eta.features import CATEGORICAL, NUMERIC


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Median delivery time per (city type, distance band); falls back to the overall median.

    This is what a dispatcher could do with a lookup table and no model.
    """

    def __init__(self, band_edges: tuple[float, ...] = (0, 3, 6, 9, 12, 50)):
        self.band_edges = band_edges

    def _keys(self, X: pd.DataFrame) -> pd.Series:
        bands = pd.cut(X["distance_km"], bins=list(self.band_edges), include_lowest=True,
                       labels=False)
        return X["city_type"].astype(str) + "|" + bands.astype("Int64").astype(str)

    def fit(self, X: pd.DataFrame, y: Any) -> MedianBaseline:
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.table_ = y.groupby(self._keys(X)).median().to_dict()
        self.global_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = self._keys(X)
        return keys.map(self.table_).fillna(self.global_).to_numpy(dtype=float)


def _encoder(scale_numeric: bool) -> ColumnTransformer:
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def make_linear() -> Pipeline:
    """Ridge regression on scaled numerics and one-hot categoricals (alpha picked by CV)."""
    return Pipeline([("prep", _encoder(True)), ("model", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def make_xgb(params: dict[str, Any], seed: int = 42) -> Pipeline:
    """Gradient-boosted trees with the given hyperparameters."""
    model = XGBRegressor(**params, random_state=seed, n_jobs=-1, tree_method="hist")
    return Pipeline([("prep", _encoder(False)), ("model", model)])


def make_quantile(alpha: float, max_iter: int = 400, seed: int = 42) -> Pipeline:
    """Gradient boosting with pinball loss for the `alpha` quantile of delivery time."""
    model = HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=max_iter,
                                          learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
    return Pipeline([("prep", _encoder(False)), ("model", model)])


def tune_xgb(X: pd.DataFrame, y: pd.Series, groups: pd.Series, base: dict[str, Any],
             n_iter: int, folds: int, seed: int = 42) -> tuple[dict[str, Any], float]:
    """Small random search over tree depth, learning rate and regularisation, scored by MAE
    with folds grouped by city so the search does not reward memorising a city."""
    space = {
        "model__max_depth": randint(3, 9),
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__min_child_weight": randint(1, 10),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_lambda": loguniform(0.1, 10),
    }
    search = RandomizedSearchCV(make_xgb(base, seed), space, n_iter=n_iter,
                                cv=GroupKFold(n_splits=folds), scoring="neg_mean_absolute_error",
                                random_state=seed, n_jobs=1)
    search.fit(X, y, groups=groups)
    best = dict(base)
    for k, v in search.best_params_.items():
        val = v.item() if hasattr(v, "item") else v
        best[k.replace("model__", "")] = round(val, 4) if isinstance(val, float) else val
    return best, float(-search.best_score_)
