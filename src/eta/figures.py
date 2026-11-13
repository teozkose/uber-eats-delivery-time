"""Figures for the README and the report page. Each stays well under 1.5 MB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 120
BLUE, GREY, ORANGE = "#2b6cb0", "#a0aec0", "#dd6b20"

SCHEME_LABELS = {"random_kfold": "Random 5-fold\n(original)",
                 "city_grouped_kfold": "Grouped by city\n5-fold",
                 "time_holdout": "Time-ordered\nholdout"}
POLICY_LABELS = {"baseline_median": "Lookup median", "model_point": "Model point ETA",
                 "model_p90": "Model 90th pct ETA"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def pred_vs_actual(frame: pd.DataFrame, path: Path, seed: int = 42) -> None:
    """Predicted vs actual minutes with each order's 80% interval on a random sample."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.hexbin(frame["y"], frame["pred"], gridsize=40, cmap="Blues", mincnt=1)
    lo, hi = float(frame[["y", "pred"]].min().min()), float(frame[["y", "pred"]].max().max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Actual delivery time (min)")
    ax.set_ylabel("Predicted (min)")
    ax.set_title("Holdout: predicted vs actual")

    ax = axes[1]
    s = frame.sample(n=min(80, len(frame)), random_state=seed).sort_values("pred").reset_index(drop=True)
    inside = (s["y"] >= s["lo"]) & (s["y"] <= s["hi"])
    ax.vlines(s.index, s["lo"], s["hi"], color=GREY, linewidth=2, label="80% interval")
    ax.scatter(s.index[inside], s["y"][inside], s=12, color=BLUE, label="actual, inside", zorder=3)
    ax.scatter(s.index[~inside], s["y"][~inside], s=14, color=ORANGE, label="actual, outside", zorder=3)
    ax.set_xlabel("80 holdout orders, sorted by prediction")
    ax.set_ylabel("Minutes")
    ax.set_title("Prediction intervals")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def validation_schemes(metrics: dict[str, Any], path: Path) -> None:
    """MAE of one model under three ways of splitting the data."""
    sch = metrics["validation_schemes"]
    keys = list(SCHEME_LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([SCHEME_LABELS[k] for k in keys], [sch[k]["mae_mean"] for k in keys],
           yerr=[sch[k]["mae_std"] for k in keys], color=[GREY, BLUE, BLUE], capsize=4)
    for i, k in enumerate(keys):
        ax.text(i, sch[k]["mae_mean"] + 0.08, f"{sch[k]['mae_mean']:.2f}", ha="center")
    ax.set_ylabel("MAE (min), fold mean ± sd")
    ax.set_title("Same model, three validation schemes")
    _save(fig, path)


def segment_errors(segs: pd.DataFrame, path: Path) -> None:
    """MAE with 95% bootstrap intervals by traffic, weather and distance band."""
    names = ["traffic", "weather", "distance_band"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, name in zip(axes, names, strict=True):
        d = segs[segs["segment"] == name].sort_values("mae")
        err = np.vstack([d["mae"] - d["mae_lo"], d["mae_hi"] - d["mae"]])
        ax.barh(d["level"], d["mae"], xerr=err, color=BLUE, capsize=3)
        ax.set_title(name.replace("_", " ").capitalize())
        ax.set_xlabel("MAE (min)")
    _save(fig, path)


def importance(perm: pd.DataFrame, shap_s: pd.Series, path: Path) -> None:
    """Permutation importance with 95% intervals next to mean |SHAP|."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    p = perm.head(12).iloc[::-1]
    axes[0].barh(p["feature"], p["mae_increase"],
                 xerr=np.vstack([p["mae_increase"] - p["lo"], p["hi"] - p["mae_increase"]]),
                 color=BLUE, capsize=3)
    axes[0].set_title("Permutation: MAE increase when shuffled")
    axes[0].set_xlabel("minutes")
    s = shap_s.head(12).iloc[::-1]
    axes[1].barh(s.index, s.values, color=GREY)
    axes[1].set_title("SHAP: mean |contribution|")
    axes[1].set_xlabel("minutes")
    _save(fig, path)


def eta_policy(metrics: dict[str, Any], path: Path) -> None:
    """Late rate against the average promised ETA for each promise policy and grace window."""
    sens = pd.DataFrame(metrics["business"]["sensitivity"])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    markers = {0: "o", 5: "s", 10: "^"}
    colors = {"baseline_median": GREY, "model_point": BLUE, "model_p90": ORANGE}
    for (pol, g), d in sens.groupby(["policy", "grace_min"]):
        ax.scatter(d["mean_promise_min"], d["late_rate"] * 100, color=colors.get(pol, "black"),
                   marker=markers.get(g, "o"), s=50)
    for pol, c in colors.items():
        ax.scatter([], [], color=c, label=POLICY_LABELS[pol])
    for g, m in markers.items():
        ax.scatter([], [], color="black", marker=m, label=f"grace {g} min")
    ax.set_xlabel("Average promised ETA (min)")
    ax.set_ylabel("Late orders (%)")
    ax.set_title("Promise longer, refund less")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def make_all(metrics: dict[str, Any], art: dict[str, Any], out: Path) -> list[Path]:
    """Write every figure into `out` and return the paths."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"{name}.png" for name in
             ["pred_vs_actual", "validation_schemes", "error_by_segment", "feature_importance",
              "eta_policy_cost"]}
    pred_vs_actual(art["frame"], paths["pred_vs_actual"])
    validation_schemes(metrics, paths["validation_schemes"])
    segment_errors(art["segments"], paths["error_by_segment"])
    importance(art["perm"], art["shap"], paths["feature_importance"])
    eta_policy(metrics, paths["eta_policy_cost"])
    return list(paths.values())
