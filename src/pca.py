"""Does compressing the features with PCA help? Retrain the same gradient-boosting model on
principal components that keep 95% of the variance and compare with the full-feature model.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.model import RESULTS, XGB_PARAMS, evaluate, load_metrics, load_split, save_metrics


def plot_variance(pca, path):
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", markersize=3)
    ax.axhline(0.95, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA on standardized features")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()

    scaler = StandardScaler().fit(X_train)
    full_pca = PCA(random_state=0).fit(scaler.transform(X_train))
    plot_variance(full_pca, os.path.join(RESULTS, "pca_variance.png"))

    pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=0), XGBRegressor(**XGB_PARAMS))
    pipe.fit(X_train, y_train)
    n_components = int(pipe.named_steps["pca"].n_components_)
    pca_metrics = evaluate(pipe, X_test, y_test)

    metrics = load_metrics()
    metrics["pca"] = {"n_components_95pct": n_components, "n_features": int(X_train.shape[1]),
                      "xgboost_on_components": pca_metrics}
    save_metrics(metrics)
    print(f"components for 95% variance: {n_components} of {X_train.shape[1]} features")
    print("xgboost on components:", pca_metrics)
    print("xgboost on raw features:", metrics["models"]["xgboost"])
