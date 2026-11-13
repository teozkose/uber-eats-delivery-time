"""Small end-to-end run on a synthetic sample generated inside the test."""

import json

from conftest import make_raw

from eta.data import RENAME, load_raw
from eta.features import build_table
from eta.pipeline import run_experiment
from eta.report import fill, md_table
from eta.validate import validate_raw


def test_pipeline_end_to_end(tmp_path, small_cfg):
    csv = tmp_path / "raw.csv"
    make_raw(n=900, seed=3).to_csv(csv, index=False)
    raw = load_raw(csv)
    assert set(RENAME.values()) <= set(raw.columns)
    summary = validate_raw(raw)
    table, log = build_table(raw, small_cfg)
    table = table.merge(raw[["order_id", "courier_id"]], on="order_id")

    metrics, art = run_experiment(table, small_cfg)
    json.dumps(metrics)  # must be JSON-ready

    h = metrics["headline"]
    assert h["mae"] < h["baseline_mae"]           # the signal is learnable
    assert h["mae_ci"][0] <= h["mae"] <= h["mae_ci"][1]
    assert set(metrics["validation_schemes"]) == {"random_kfold", "city_grouped_kfold", "time_holdout"}
    assert 0 <= metrics["intervals"]["coverage"] <= 1
    assert set(metrics["business"]["policies"]) == {"baseline_median", "model_point", "model_p90"}
    assert len(art["frame"]) == metrics["split"]["test_rows"]

    metrics["data"] = {**summary, "cleaning": log, "modelling_rows": len(table)}
    metrics["split"].update({"courier_overlap_time_holdout": 1.0, "couriers_per_city": 10.0})
    text = fill("MAE {{MAE}} ({{MAE_CI}})\n{{MODEL_TABLE}}", metrics, md_table)
    assert "{{" not in text and "| Model |" in text
