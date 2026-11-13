"""End-to-end pipeline. Steps: data (download), features (validate + build table), run (models,
evaluation, figures, metrics.json), report (README + docs page).

Usage: python -m eta.pipeline {data,features,run,report,all}
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from eta import evaluate as ev
from eta.config import load_config, resolve
from eta.data import TARGET, download, load_raw
from eta.features import CATEGORICAL, FEATURES, NUMERIC, build_table, time_split
from eta.models import MedianBaseline, make_linear, make_quantile, make_xgb, tune_xgb
from eta.validate import validate_raw, validate_table


def courier_overlap(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Share of test orders whose courier also delivered in the training rows."""
    if "courier_id" not in train.columns:
        return float("nan")
    return float(test["courier_id"].isin(set(train["courier_id"])).mean())


def compare_validation(table: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Score the same gradient-boosting setup under three splits: random k-fold (the original
    write-up), k-fold grouped by city, and the time-ordered holdout."""
    v, xgb_params = cfg["validation"], cfg["models"]["xgboost"]
    seed = v["seed"]
    X, y = table[FEATURES], table[TARGET]

    def factory() -> Any:
        return make_xgb(xgb_params, seed)

    random_kf = KFold(n_splits=v["random_folds"], shuffle=True, random_state=seed)
    grouped = GroupKFold(n_splits=min(v["group_folds"], table["city_code"].nunique()))
    out = {
        "random_kfold": ev.cross_validate(factory, X, y, random_kf),
        "city_grouped_kfold": ev.cross_validate(factory, X, y, grouped, table["city_code"]),
    }
    train, test = time_split(table, v["holdout_start"])
    pred = factory().fit(train[FEATURES], train[TARGET]).predict(test[FEATURES])
    out["time_holdout"] = {"mae_mean": round(float(np.mean(np.abs(pred - test[TARGET]))), 3),
                           "mae_std": 0.0, "r2_mean": ev.point_metrics(test[TARGET], pred)["r2"],
                           "r2_std": 0.0, "folds": 1}
    return out


def run_experiment(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit every model on the time-ordered training window and evaluate on the holdout.

    Returns (metrics, artifacts): metrics is JSON-ready; artifacts holds frames for figures.
    """
    v, e, b = cfg["validation"], cfg["evaluation"], cfg["business"]
    seed, n_boot, off_by = v["seed"], e["n_boot"], e["off_by_minutes"]
    train, test = time_split(table, v["holdout_start"])
    if len(train) == 0 or len(test) == 0:
        raise ValueError("time split left an empty side; check validation.holdout_start")
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    metrics: dict[str, Any] = {
        "split": {"train_rows": int(len(train)), "test_rows": int(len(test)),
                  "train_dates": [str(train.order_date.min().date()), str(train.order_date.max().date())],
                  "test_dates": [str(test.order_date.min().date()), str(test.order_date.max().date())],
                  "cities": int(table["city_code"].nunique()),
                  "target_mean_min": round(float(table[TARGET].mean()), 2),
                  "target_std_min": round(float(table[TARGET].std()), 2)},
    }

    metrics["validation_schemes"] = compare_validation(table, cfg)

    tuned_params, tuned_cv_mae = tune_xgb(Xtr, ytr, train["city_code"], cfg["models"]["xgboost"],
                                          cfg["models"]["tuning"]["n_iter"],
                                          cfg["models"]["tuning"]["folds"], seed)
    metrics["tuning"] = {"best_params": tuned_params, "grouped_cv_mae": round(tuned_cv_mae, 3),
                         "n_iter": cfg["models"]["tuning"]["n_iter"]}

    models = {
        "median_baseline": MedianBaseline(tuple(cfg["features"]["distance_bands_km"])),
        "ridge": make_linear(),
        "xgboost_default": make_xgb(cfg["models"]["xgboost"], seed),
        "xgboost_tuned": make_xgb(tuned_params, seed),
    }
    preds: dict[str, np.ndarray] = {}
    metrics["models"] = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        preds[name] = model.predict(Xte)
        metrics["models"][name] = ev.metrics_with_ci(yte, preds[name], n_boot, seed, off_by)
        print(f"{name:18} MAE {metrics['models'][name]['mae']:.2f}  R2 {metrics['models'][name]['r2']:.3f}")
    best = min(["xgboost_default", "xgboost_tuned"], key=lambda k: metrics["models"][k]["mae"])
    metrics["best_model"] = best
    final, pred = models[best], preds[best]
    base_mae = metrics["models"]["median_baseline"]["mae"]
    metrics["headline"] = {
        "model": best, "mae": metrics["models"][best]["mae"],
        "mae_ci": metrics["models"][best]["mae_ci"], "r2": metrics["models"][best]["r2"],
        "off_by_10": metrics["models"][best]["off_by_10"],
        "off_by_10_ci": metrics["models"][best]["off_by_10_ci"],
        "baseline_mae": base_mae,
        "mae_reduction_vs_baseline": round(1 - metrics["models"][best]["mae"] / base_mae, 4),
    }

    # 80% prediction intervals from two quantile models
    iv = cfg["models"]["interval"]
    lower = make_quantile(iv["lower"], iv["max_iter"], seed).fit(Xtr, ytr).predict(Xte)
    upper = make_quantile(iv["upper"], iv["max_iter"], seed).fit(Xtr, ytr).predict(Xte)
    upper = np.maximum(upper, lower)
    frame = test[["traffic", "weather", "distance_band", "city_type", "city_code"]].copy()
    frame["y"], frame["pred"], frame["lo"], frame["hi"] = yte.to_numpy(), pred, lower, upper
    cov = ev.interval_coverage(yte, lower, upper)
    cov["nominal"] = round(iv["upper"] - iv["lower"], 2)
    cov["by_traffic"] = {lvl: ev.interval_coverage(g["y"], g["lo"], g["hi"])["coverage"]
                         for lvl, g in frame.groupby("traffic")}
    metrics["intervals"] = cov

    # errors by segment
    segs = pd.concat([ev.segment_errors(frame, s, off_by=off_by, seed=seed)
                      for s in ["traffic", "weather", "distance_band"]], ignore_index=True)
    metrics["segments"] = segs.to_dict(orient="records")

    # what drives it
    sample = test.sample(n=min(5000, len(test)), random_state=seed)
    perm = ev.permutation_with_ci(final, sample[FEATURES], sample[TARGET], e["permutation_repeats"], seed)
    shap_rows = test.sample(n=min(e["shap_rows"], len(test)), random_state=seed)
    shap_s = ev.shap_by_feature(final, shap_rows[FEATURES], NUMERIC, CATEGORICAL)
    top_perm = list(perm["feature"].head(5))
    metrics["importance"] = {
        "permutation": [{"feature": r.feature, "mae_increase": round(r.mae_increase, 3),
                         "ci": [round(r.lo, 3), round(r.hi, 3)]} for r in perm.itertuples()],
        "shap": [{"feature": f, "mean_abs_shap_min": round(float(val), 3)} for f, val in shap_s.items()],
        "top5_overlap": len(set(top_perm) & set(shap_s.head(5).index)),
    }

    # business framing: what an ETA promise costs in refunds
    policies = {"baseline_median": preds["median_baseline"], "model_point": pred, "model_p90": upper}
    biz: dict[str, Any] = {"refund": b["refund_per_late_order"], "grace_min": b["grace_minutes"],
                           "policies": {}, "sensitivity": []}
    for name, promise in policies.items():
        row = ev.eta_cost(yte, promise, b["grace_minutes"], b["refund_per_late_order"])
        row["cost_ci"] = ev.eta_cost_ci(yte, promise, b["grace_minutes"], b["refund_per_late_order"],
                                        n_boot, seed)
        biz["policies"][name] = row
        for g in b["grace_sensitivity"]:
            s = ev.eta_cost(yte, promise, g, b["refund_per_late_order"])
            biz["sensitivity"].append({"policy": name, "grace_min": g, **s})
    metrics["business"] = biz

    artifacts = {"frame": frame, "perm": perm, "shap": shap_s, "segments": segs}
    return metrics, artifacts


def step_data(cfg: dict[str, Any]) -> None:
    download(cfg["data"]["url"], resolve(cfg["data"]["raw"]), cfg["data"]["min_raw_bytes"])


def step_features(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = load_raw(resolve(cfg["data"]["raw"]))
    summary = validate_raw(raw)
    table, log = build_table(raw, cfg)
    validate_table(table, NUMERIC, CATEGORICAL, cfg["cleaning"]["max_distance_km"])
    table = table.merge(raw[["order_id", "courier_id"]], on="order_id", how="left")
    out = resolve(cfg["data"]["table"])
    table.to_parquet(out, index=False)
    summary["cleaning"] = log
    summary["modelling_rows"] = int(len(table))
    results = resolve(cfg["paths"]["results"])
    results.mkdir(exist_ok=True)
    (results / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"raw {summary['rows']:,} rows -> {len(table):,} modelling rows")
    return summary


def step_run(cfg: dict[str, Any]) -> dict[str, Any]:
    from eta.figures import make_all

    table = pd.read_parquet(resolve(cfg["data"]["table"]))
    metrics, art = run_experiment(table, cfg)
    train, test = time_split(table, cfg["validation"]["holdout_start"])
    metrics["split"]["courier_overlap_time_holdout"] = round(courier_overlap(train, test), 4)
    shuffled = table.sample(frac=1.0, random_state=cfg["validation"]["seed"])
    cut = int(len(shuffled) * 0.8)
    metrics["split"]["courier_overlap_random_split"] = round(
        courier_overlap(shuffled.iloc[:cut], shuffled.iloc[cut:]), 4)
    metrics["split"]["couriers_per_city"] = round(
        float(table.groupby("city_code")["courier_id"].nunique().mean()), 1)
    results = resolve(cfg["paths"]["results"])
    summary_path = results / "data_summary.json"
    if summary_path.exists():
        metrics["data"] = json.loads(summary_path.read_text())
    make_all(metrics, art, results)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))
    art["segments"].to_csv(results / "segment_errors.csv", index=False)
    print(f"saved {results / 'metrics.json'}")
    return metrics


def step_report(cfg: dict[str, Any]) -> None:
    from eta.report import render_all

    render_all(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["data", "features", "run", "report", "all"])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    steps = ["data", "features", "run", "report"] if args.step == "all" else [args.step]
    for s in steps:
        {"data": step_data, "features": step_features, "run": step_run, "report": step_report}[s](cfg)


if __name__ == "__main__":
    main()
