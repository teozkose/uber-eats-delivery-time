"""Synthetic delivery records in the raw file's shape, so tests never touch the network."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from eta.config import load_config
from eta.data import RENAME

CITIES = ["INDO", "BANG", "MUM", "PUNE", "HYD", "CHEN"]


def make_raw(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """Raw-format frame (original column names) with a known signal in distance and traffic."""
    rng = np.random.default_rng(seed)
    city = rng.choice(CITIES, n)
    courier = [f"{c}RES{rng.integers(1, 20):02d}DEL0{rng.integers(1, 3)}" for c in city]
    lat = rng.uniform(12, 28, n)
    lon = rng.uniform(72, 88, n)
    dlat = lat + rng.uniform(0.005, 0.1, n)
    dlon = lon + rng.uniform(0.005, 0.1, n)
    day = rng.integers(1, 21, n)
    hour = rng.integers(8, 23, n)
    ordered = pd.to_datetime("2022-03-01") + pd.to_timedelta(day - 1, "D") + pd.to_timedelta(hour, "h")
    picked = ordered + pd.to_timedelta(rng.choice([5, 10, 15], n), "m")
    traffic = rng.choice(["Low", "Medium", "High", "Jam"], n)
    weather = rng.choice(["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"], n)
    km = np.sqrt((dlat - lat) ** 2 + (dlon - lon) ** 2) * 111
    jam = pd.Series(traffic).map({"Low": 0, "Medium": 3, "High": 6, "Jam": 10}).to_numpy()
    minutes = np.clip(12 + 1.2 * km + jam + rng.normal(0, 2, n), 10, 60).round()
    raw = pd.DataFrame({
        "ID": [f"0x{i:05x}" for i in range(n)],
        "Delivery_person_ID": courier,
        "Delivery_person_Age": rng.integers(20, 40, n).astype(float),
        "Delivery_person_Ratings": rng.uniform(3.5, 5, n).round(1),
        "Restaurant_latitude": lat, "Restaurant_longitude": lon,
        "Delivery_location_latitude": dlat, "Delivery_location_longitude": dlon,
        "Order_Date": ordered.strftime("%d-%m-%Y"),
        "Time_Orderd": ordered.strftime("%Y-%m-%d %H:%M:%S"),
        "Time_Order_picked": picked.strftime("%Y-%m-%d %H:%M:%S"),
        "Weatherconditions": weather, "Road_traffic_density": traffic,
        "Vehicle_condition": rng.integers(0, 3, n).astype(float),
        "Type_of_order": rng.choice(["Snack", "Meal", "Drinks", "Buffet"], n),
        "Type_of_vehicle": rng.choice(["motorcycle", "scooter"], n),
        "multiple_deliveries": rng.integers(0, 3, n).astype(float),
        "Festival": rng.choice(["No", "Yes"], n, p=[0.95, 0.05]),
        "City": rng.choice(["Metropolitian", "Urban"], n),
        "Time_taken(min)": minutes,
    })
    return raw


@pytest.fixture
def raw() -> pd.DataFrame:
    """Renamed raw frame, as returned by eta.data.load_raw."""
    return make_raw().rename(columns=RENAME)


@pytest.fixture
def small_cfg() -> dict:
    """The project config shrunk for a fast end-to-end run."""
    cfg = copy.deepcopy(load_config())
    cfg["validation"].update({"holdout_start": "2022-03-15", "random_folds": 3, "group_folds": 3})
    cfg["models"]["xgboost"]["n_estimators"] = 40
    cfg["models"]["tuning"].update({"n_iter": 2, "folds": 2})
    cfg["models"]["interval"]["max_iter"] = 40
    cfg["evaluation"].update({"n_boot": 50, "permutation_repeats": 3, "shap_rows": 100})
    return cfg
