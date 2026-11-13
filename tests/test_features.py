import numpy as np
import pandas as pd
import pytest

from eta.data import TARGET
from eta.features import (
    FEATURES,
    build_table,
    city_code,
    distance_band,
    haversine_km,
    pickup_delay_minutes,
    time_split,
)


def test_haversine_known_distance():
    # Mumbai to Pune is about 120 km in a straight line
    d = haversine_km(19.0760, 72.8777, 18.5204, 73.8567)
    assert 115 < d < 125
    assert haversine_km(10.0, 70.0, 10.0, 70.0) == pytest.approx(0.0)


def test_city_code_from_courier_id():
    ids = pd.Series(["INDORES13DEL02", "BANGRES18DEL01", "garbage"])
    assert city_code(ids).tolist() == ["INDO", "BANG", "UNK"]


def test_pickup_delay_wraps_past_midnight():
    ordered = pd.to_datetime(pd.Series(["2022-03-01 23:55:00", "2022-03-01 12:00:00"]))
    picked = pd.to_datetime(pd.Series(["2022-03-01 00:05:00", "2022-03-01 12:15:00"]))
    assert pickup_delay_minutes(ordered, picked).tolist() == [10.0, 15.0]


def test_distance_band_labels():
    bands = distance_band(pd.Series([1.0, 4.0, 20.0]), [0, 3, 6, 50])
    assert bands.tolist() == ["0-3 km", "3-6 km", "6+ km"]


def test_build_table_drops_bad_coordinates_and_logs(raw, small_cfg):
    raw = raw.copy()
    raw.loc[0:9, "rest_lat"] = 0.0
    table, log = build_table(raw, small_cfg)
    assert log[0]["dropped"] == 10
    assert len(table) == len(raw) - 10
    assert table[FEATURES].isna().sum().sum() == 0
    assert (table["distance_km"] <= small_cfg["cleaning"]["max_distance_km"]).all()
    assert table["order_date"].is_monotonic_increasing


def test_time_split_is_ordered(raw, small_cfg):
    table, _ = build_table(raw, small_cfg)
    train, test = time_split(table, "2022-03-15")
    assert train["order_date"].max() < test["order_date"].min()
    assert len(train) + len(test) == len(table)
    assert np.isfinite(test[TARGET]).all()
