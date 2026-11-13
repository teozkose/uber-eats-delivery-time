"""Turn validated raw records into a modelling table, logging how many rows each cleaning step drops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eta.data import TARGET

NUMERIC = [
    "courier_age", "courier_rating", "distance_km", "vehicle_condition", "multiple_deliveries",
    "pickup_delay_min", "order_hour", "is_weekend", "is_peak_hour",
]
CATEGORICAL = ["weather", "traffic", "order_type", "vehicle_type", "festival", "city_type"]
FEATURES = NUMERIC + CATEGORICAL
META = ["order_id", "order_date", "city_code", "distance_band"]


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Great-circle distance in km between coordinate pairs (scalars or arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def city_code(courier_id: pd.Series) -> pd.Series:
    """Courier ids look like INDORES13DEL02: the letters before 'RES' name the city."""
    return courier_id.astype(str).str.extract(r"^([A-Z]+?)RES", expand=False).fillna("UNK")


def pickup_delay_minutes(ordered: pd.Series, picked: pd.Series) -> pd.Series:
    """Minutes from order to pickup; a pickup after midnight wraps around the day."""
    delay = (picked - ordered).dt.total_seconds() / 60
    return delay.where(delay >= 0, delay + 24 * 60)


def distance_band(distance_km: pd.Series, edges: list[float]) -> pd.Series:
    """Label distances with the configured band edges, e.g. '3-6 km'."""
    labels = [f"{int(lo)}-{int(hi)} km" if hi < edges[-1] else f"{int(lo)}+ km"
              for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return pd.cut(distance_km, bins=edges, labels=labels, include_lowest=True).astype(str)


def build_table(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Clean the raw frame and derive features. Returns the table and a per-step row-count log."""
    clean = cfg["cleaning"]
    log: list[dict[str, Any]] = []
    df = raw.copy()

    def step(name: str, mask: pd.Series) -> None:
        nonlocal df
        before = len(df)
        df = df[mask]
        log.append({"step": name, "rows_before": before, "rows_after": len(df),
                    "dropped": before - len(df)})

    for c in ["rest_lat", "rest_lon", "drop_lat", "drop_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").abs()
    step("drop zero restaurant/drop-off coordinates",
         (df["rest_lat"] > 1) & (df["drop_lat"] > 1) & (df["rest_lon"] > 1) & (df["drop_lon"] > 1))

    df["distance_km"] = haversine_km(df["rest_lat"], df["rest_lon"], df["drop_lat"], df["drop_lon"])
    step(f"drop trips over {clean['max_distance_km']} km", df["distance_km"] <= clean["max_distance_km"])

    ordered = pd.to_datetime(df["time_ordered"], errors="coerce")
    picked = pd.to_datetime(df["time_picked"], errors="coerce")
    df["pickup_delay_min"] = pickup_delay_minutes(ordered, picked)
    df["order_hour"] = ordered.dt.hour
    step("drop rows with unparseable order or pickup times",
         df["pickup_delay_min"].notna() & df["order_hour"].notna())
    step(f"drop pickup delays over {clean['max_pickup_delay_min']} min",
         df["pickup_delay_min"] <= clean["max_pickup_delay_min"])

    df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y")
    df["is_weekend"] = df["order_date"].dt.day_of_week.isin([5, 6]).astype(float)
    df["is_peak_hour"] = df["order_hour"].isin(clean["peak_hours"]).astype(float)
    df["order_hour"] = df["order_hour"].astype(float)
    df["city_code"] = city_code(df["courier_id"])
    df["distance_band"] = distance_band(df["distance_km"], cfg["features"]["distance_bands_km"])
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    for c in ["courier_age", "courier_rating", "vehicle_condition", "multiple_deliveries", TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    table = df[META + FEATURES + [TARGET]].sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return table, log


def time_split(table: pd.DataFrame, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orders before `start` train, orders on or after it test."""
    cut = pd.Timestamp(start)
    return table[table["order_date"] < cut].copy(), table[table["order_date"] >= cut].copy()
