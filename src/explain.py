"""Explain the gradient-boosting model two ways and check that they agree.

1. SHAP values on a test sample: average absolute contribution of each feature.
2. Permutation importance on the full test set: drop in R² when a feature is shuffled.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

from src.model import MODELS, RESULTS, SEED, load_metrics, load_split, save_metrics

TOP_N = 12


def shap_importance(model, X_sample) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)
    return pd.Series(np.abs(values).mean(axis=0), index=X_sample.columns).sort_values(ascending=False)


def perm_importance(model, X_test, y_test) -> pd.Series:
    result = permutation_importance(model, X_test, y_test, scoring="r2", n_repeats=5,
                                    random_state=SEED, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)


def plot_two_rankings(shap_s, perm_s, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, s, title, xlabel in [
        (axes[0], shap_s.head(TOP_N)[::-1], "SHAP: mean |contribution| (min)", "minutes"),
        (axes[1], perm_s.head(TOP_N)[::-1], "Permutation: drop in R² when shuffled", "ΔR²"),
    ]:
        ax.barh(s.index, s.values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    model = joblib.load(os.path.join(MODELS, "xgboost.joblib"))

    sample = X_test.sample(n=min(3000, len(X_test)), random_state=SEED)
    shap_s = shap_importance(model, sample)
    perm_s = perm_importance(model, X_test, y_test)
    plot_two_rankings(shap_s, perm_s, os.path.join(RESULTS, "feature_importance.png"))

    top_shap = list(shap_s.head(TOP_N).index)
    top_perm = list(perm_s.head(TOP_N).index)
    overlap = len(set(top_shap[:5]) & set(top_perm[:5]))

    metrics = load_metrics()
    metrics["importance"] = {
        "shap_top": [{"feature": f, "mean_abs_shap_min": round(float(v), 3)} for f, v in shap_s.head(TOP_N).items()],
        "permutation_top": [{"feature": f, "r2_drop": round(float(v), 4)} for f, v in perm_s.head(TOP_N).items()],
        "top5_overlap": overlap,
        "shap_sample_rows": int(len(sample)),
    }
    save_metrics(metrics)
    print("SHAP top 5:", top_shap[:5])
    print("Permutation top 5:", top_perm[:5])
    print(f"overlap in top 5: {overlap}/5")
