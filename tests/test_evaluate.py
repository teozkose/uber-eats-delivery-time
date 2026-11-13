import numpy as np
import pandas as pd
import pytest

from eta.evaluate import (
    bootstrap_ci,
    eta_cost,
    interval_coverage,
    point_metrics,
    segment_errors,
    share_off_by,
)
from eta.models import MedianBaseline


def test_point_metrics_simple_case():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 18.0, 30.0, 52.0])
    m = point_metrics(y, pred)
    assert m["mae"] == pytest.approx(4.0)
    assert m["off_by_10"] == pytest.approx(0.25)
    assert share_off_by(y, y) == 0.0


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.normal(30, 5, 500)
    pred = y + rng.normal(0, 2, 500)
    est = float(np.mean(np.abs(y - pred)))
    lo, hi = bootstrap_ci(y, pred, lambda a, b: float(np.mean(np.abs(a - b))), n_boot=300)
    assert lo < est < hi
    assert hi - lo < 0.5


def test_interval_coverage_counts():
    cov = interval_coverage([5, 10, 15, 20], [4, 11, 10, 10], [6, 12, 20, 19])
    assert cov["coverage"] == 0.5
    assert cov["below"] == 0.25 and cov["above"] == 0.25


def test_eta_cost_late_rule():
    out = eta_cost([20, 26, 31, 40], [25, 25, 25, 25], grace_min=5, refund=4.0)
    assert out["late_rate"] == 0.5        # 31 and 40 are more than 5 min past 25
    assert out["cost_per_1000"] == 2000.0
    assert out["mean_promise_min"] == 25.0


def test_segment_errors_shape():
    frame = pd.DataFrame({"traffic": ["Low", "Low", "Jam", "Jam"],
                          "y": [10.0, 12.0, 30.0, 40.0], "pred": [11.0, 12.0, 20.0, 41.0]})
    out = segment_errors(frame, "traffic", n_boot=20)
    assert list(out["level"]) == ["Jam", "Low"]       # sorted worst first
    assert out.loc[0, "mae"] == pytest.approx(5.5)


def test_median_baseline_falls_back_to_global():
    X = pd.DataFrame({"city_type": ["Urban", "Urban", "Metropolitian"], "distance_km": [1.0, 2.0, 10.0]})
    model = MedianBaseline((0, 3, 50)).fit(X, [20.0, 30.0, 40.0])
    new = pd.DataFrame({"city_type": ["Urban", "Semi-Urban"], "distance_km": [1.5, 1.0]})
    assert model.predict(new).tolist() == [25.0, 30.0]
