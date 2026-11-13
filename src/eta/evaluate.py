"""Metrics, bootstrap intervals, validation schemes, segment errors and the ETA cost model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

Metric = Callable[[np.ndarray, np.ndarray], float]


def share_off_by(y: Any, pred: Any, minutes: float = 10) -> float:
    """Fraction of orders where the prediction misses by `minutes` or more, either direction."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred)) >= minutes))


def point_metrics(y: Any, pred: Any, off_by: float = 10) -> dict[str, float]:
    """MAE, RMSE, R² and the share of orders off by `off_by`+ minutes."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(y, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
        "r2": round(float(r2_score(y, pred)), 4),
        "off_by_10": round(share_off_by(y, pred, off_by), 4),
    }


def bootstrap_ci(y: Any, pred: Any, metric: Metric, n_boot: int = 1000, seed: int = 42,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for `metric(y, pred)` resampling orders with replacement."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = metric(y[idx], pred[idx])
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return float(lo), float(hi)


def metrics_with_ci(y: Any, pred: Any, n_boot: int, seed: int, off_by: float = 10) -> dict[str, Any]:
    """Point metrics plus 95% bootstrap intervals for MAE, R² and the off-by-N share."""
    out: dict[str, Any] = point_metrics(y, pred, off_by)
    fns: dict[str, Metric] = {
        "mae": lambda a, b: float(np.mean(np.abs(a - b))),
        "r2": lambda a, b: float(r2_score(a, b)),
        "off_by_10": lambda a, b: share_off_by(a, b, off_by),
    }
    for name, fn in fns.items():
        lo, hi = bootstrap_ci(y, pred, fn, n_boot, seed)
        out[f"{name}_ci"] = [round(lo, 4), round(hi, 4)]
    return out


def cross_validate(factory: Callable[[], Any], X: pd.DataFrame, y: pd.Series,
                   splitter: KFold | GroupKFold, groups: pd.Series | None = None) -> dict[str, Any]:
    """Fit a fresh model per fold; return mean and spread of fold MAE and R²."""
    maes, r2s = [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = factory().fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        r2s.append(r2_score(y.iloc[test_idx], pred))
    return {"mae_mean": round(float(np.mean(maes)), 3), "mae_std": round(float(np.std(maes)), 3),
            "r2_mean": round(float(np.mean(r2s)), 4), "r2_std": round(float(np.std(r2s)), 4),
            "folds": len(maes)}


def interval_coverage(y: Any, lower: Any, upper: Any) -> dict[str, float]:
    """Share of outcomes inside [lower, upper] and the average interval width."""
    y, lower, upper = (np.asarray(a, dtype=float) for a in (y, lower, upper))
    inside = (y >= lower) & (y <= upper)
    return {"coverage": round(float(inside.mean()), 4),
            "below": round(float((y < lower).mean()), 4),
            "above": round(float((y > upper).mean()), 4),
            "mean_width_min": round(float(np.mean(upper - lower)), 3)}


def segment_errors(frame: pd.DataFrame, segment: str, y: str = "y", pred: str = "pred",
                   off_by: float = 10, n_boot: int = 300, seed: int = 42) -> pd.DataFrame:
    """MAE (with a bootstrap CI), mean bias and off-by-N share for each level of `segment`."""
    rows = []
    for level, g in frame.groupby(segment, observed=True):
        err = g[pred] - g[y]
        lo, hi = bootstrap_ci(g[y], g[pred], lambda a, b: float(np.mean(np.abs(a - b))),
                              n_boot, seed)
        rows.append({"segment": segment, "level": str(level), "n": int(len(g)),
                     "mae": round(float(err.abs().mean()), 3),
                     "mae_lo": round(lo, 3), "mae_hi": round(hi, 3),
                     "bias": round(float(err.mean()), 3),
                     "off_by_10": round(float((err.abs() >= off_by).mean()), 4)})
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def eta_cost(y: Any, promised: Any, grace_min: float, refund: float) -> dict[str, float]:
    """A delivery is late when it lands more than `grace_min` after the promised ETA;
    each late order costs `refund`. Returns late rate, cost per 1,000 orders and mean promise."""
    y, promised = np.asarray(y, dtype=float), np.asarray(promised, dtype=float)
    late = y > promised + grace_min
    return {"late_rate": round(float(late.mean()), 4),
            "cost_per_1000": round(float(late.mean() * refund * 1000), 1),
            "mean_promise_min": round(float(promised.mean()), 2)}


def eta_cost_ci(y: Any, promised: Any, grace_min: float, refund: float, n_boot: int,
                seed: int) -> list[float]:
    """Bootstrap 95% interval for cost per 1,000 orders."""
    lo, hi = bootstrap_ci(y, promised,
                          lambda a, b: float(np.mean(a > b + grace_min) * refund * 1000),
                          n_boot, seed)
    return [round(lo, 1), round(hi, 1)]


def permutation_with_ci(model: Any, X: pd.DataFrame, y: pd.Series, repeats: int,
                        seed: int) -> pd.DataFrame:
    """Permutation importance (increase in MAE when a column is shuffled) per original feature,
    with a 95% interval from the spread across repeats."""
    res = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                                 n_repeats=repeats, random_state=seed, n_jobs=1)
    imp = res.importances  # rows = features, cols = repeats; positive = MAE rises
    mean = imp.mean(axis=1)
    half = 1.96 * imp.std(axis=1, ddof=1) / np.sqrt(repeats)
    out = pd.DataFrame({"feature": X.columns, "mae_increase": mean,
                        "lo": mean - half, "hi": mean + half})
    return out.sort_values("mae_increase", ascending=False).reset_index(drop=True)


def shap_by_feature(pipeline: Any, X: pd.DataFrame, numeric: list[str],
                    categorical: list[str]) -> pd.Series:
    """Mean |SHAP| (minutes) per original feature: one-hot columns are summed back to their source."""
    import shap  # optional heavy import, only needed here

    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())
    values = shap.TreeExplainer(model).shap_values(Xt)
    per_col = pd.DataFrame(values, columns=names)
    source = {}
    for name in names:
        kind, col = name.split("__", 1)
        if kind == "num":
            source[name] = col
        else:
            source[name] = next(c for c in categorical if col.startswith(c + "_"))
    grouped = per_col.T.groupby(pd.Series(source)).sum().T
    return grouped.abs().mean().reindex(numeric + categorical).sort_values(ascending=False)
