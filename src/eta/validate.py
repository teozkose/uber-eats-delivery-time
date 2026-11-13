"""Schema, range, null and duplicate checks. Each check fails loudly with every problem listed."""

from __future__ import annotations

from typing import Any

import pandas as pd

from eta.data import TARGET


class DataValidationError(ValueError):
    """Raised when a frame breaks one or more data rules."""


REQUIRED_RAW = [
    "order_id", "courier_id", "courier_age", "courier_rating", "rest_lat", "rest_lon",
    "drop_lat", "drop_lon", "order_date", "time_ordered", "time_picked", "weather", "traffic",
    "vehicle_condition", "order_type", "vehicle_type", "multiple_deliveries", "festival",
    "city_type", TARGET,
]

RANGES = {
    "courier_age": (15, 70),
    "courier_rating": (1, 5),
    "vehicle_condition": (0, 3),
    "multiple_deliveries": (0, 5),
    TARGET: (1, 180),
}

DOMAINS = {
    "traffic": {"Low", "Medium", "High", "Jam"},
    "weather": {"Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"},
    "city_type": {"Metropolitian", "Urban", "Semi-Urban"},
    "festival": {"Yes", "No"},
}


def _range_problems(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> list[str]:
    problems = []
    for col, (lo, hi) in ranges.items():
        values = pd.to_numeric(df[col], errors="coerce")
        bad = int(((values < lo) | (values > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    return problems


def validate_raw(df: pd.DataFrame) -> dict[str, Any]:
    """Check the renamed raw frame. Returns a summary dict; raises DataValidationError on failure."""
    missing = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    problems: list[str] = []
    nulls = df[REQUIRED_RAW].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {int(n)} nulls")

    dupes = int(df["order_id"].duplicated().sum())
    if dupes:
        problems.append(f"order_id: {dupes} duplicate ids")

    problems += _range_problems(df, RANGES)
    for col in ["rest_lat", "drop_lat"]:
        if (pd.to_numeric(df[col], errors="coerce").abs() > 90).any():
            problems.append(f"{col}: latitude beyond +/-90")
    for col in ["rest_lon", "drop_lon"]:
        if (pd.to_numeric(df[col], errors="coerce").abs() > 180).any():
            problems.append(f"{col}: longitude beyond +/-180")

    for col, allowed in DOMAINS.items():
        unknown = set(df[col].astype(str).str.strip()) - allowed
        if unknown:
            problems.append(f"{col}: unexpected values {sorted(unknown)[:5]}")

    dates = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        problems.append(f"order_date: {bad_dates} values not in DD-MM-YYYY form")

    if problems:
        raise DataValidationError("raw data failed validation:\n  " + "\n  ".join(problems))

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "couriers": int(df["courier_id"].nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "order_days": int(dates.nunique()),
        "zero_or_negative_coords": int(
            ((df[["rest_lat", "rest_lon", "drop_lat", "drop_lon"]] <= 0).any(axis=1)).sum()
        ),
    }


def validate_table(table: pd.DataFrame, numeric: list[str], categorical: list[str],
                   max_distance_km: float) -> None:
    """Check the modelling table: no nulls in features, sane target and distance."""
    problems = []
    cols = numeric + categorical + [TARGET, "city_code", "order_date"]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise DataValidationError(f"table is missing columns: {missing}")
    nulls = table[cols].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {int(n)} nulls after cleaning")
    if (table["distance_km"] < 0).any() or (table["distance_km"] > max_distance_km).any():
        problems.append("distance_km outside the cleaning window")
    if (table[TARGET] <= 0).any():
        problems.append(f"{TARGET}: non-positive values")
    if len(table) == 0:
        problems.append("table is empty")
    if problems:
        raise DataValidationError("feature table failed validation:\n  " + "\n  ".join(problems))
