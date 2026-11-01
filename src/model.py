"""Train the delivery-time models and write results/metrics.json.

Models: a linear-regression baseline, a random forest, and a gradient-boosting regressor (XGBoost).
Same 80/20 split for all of them.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.prepare import OUT as FEATURES, TARGET, feature_columns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
METRICS = os.path.join(RESULTS, "metrics.json")
SEED = 42

XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, random_state=SEED, n_jobs=-1)


def load_split():
    table = pd.read_parquet(FEATURES)
    cols = feature_columns(table)
    X, y = table[cols], table[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
    }


def load_metrics() -> dict:
    return json.load(open(METRICS)) if os.path.exists(METRICS) else {}


def save_metrics(m: dict):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(m, open(METRICS, "w"), indent=2)


def plot_pred_vs_actual(y_test, pred, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, pred, s=4, alpha=0.25)
    lo, hi = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Actual delivery time (min)")
    ax.set_ylabel("Predicted delivery time (min)")
    ax.set_title("Gradient boosting: predicted vs. actual (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "xgboost": XGBRegressor(**XGB_PARAMS),
    }
    metrics = load_metrics()
    metrics["dataset"] = {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                          "n_features": int(X_train.shape[1]),
                          "target_mean_min": round(float(y_train.mean()), 2),
                          "target_std_min": round(float(y_train.std()), 2)}
    metrics["models"] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics["models"][name] = evaluate(model, X_test, y_test)
        print(f"{name:18} {metrics['models'][name]}")

    xgb = models["xgboost"]
    joblib.dump(xgb, os.path.join(MODELS, "xgboost.joblib"))
    plot_pred_vs_actual(y_test, xgb.predict(X_test), os.path.join(RESULTS, "pred_vs_actual.png"))
    save_metrics(metrics)
    print(f"saved {METRICS}")
