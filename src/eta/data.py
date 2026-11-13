"""Download and load the raw delivery records."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

TARGET = "time_taken_min"

RENAME = {
    "Delivery_person_ID": "courier_id",
    "Delivery_person_Age": "courier_age",
    "Delivery_person_Ratings": "courier_rating",
    "Restaurant_latitude": "rest_lat",
    "Restaurant_longitude": "rest_lon",
    "Delivery_location_latitude": "drop_lat",
    "Delivery_location_longitude": "drop_lon",
    "Order_Date": "order_date",
    "Time_Orderd": "time_ordered",
    "Time_Order_picked": "time_picked",
    "Weatherconditions": "weather",
    "Road_traffic_density": "traffic",
    "Vehicle_condition": "vehicle_condition",
    "Type_of_order": "order_type",
    "Type_of_vehicle": "vehicle_type",
    "multiple_deliveries": "multiple_deliveries",
    "Festival": "festival",
    "City": "city_type",
    "Time_taken(min)": TARGET,
    "ID": "order_id",
}

RAW_COLUMNS = list(RENAME)


def download(url: str, dest: Path, min_bytes: int = 1_000_000) -> Path:
    """Fetch the CSV once; skip if a file of plausible size is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"already present: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    size = dest.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"download looks truncated: {size} bytes")
    print(f"saved {dest.name} ({size / 1e6:.1f} MB)")
    return dest


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV, keep the columns the project uses and give them short names."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"raw file is missing columns: {missing}")
    return df[RAW_COLUMNS].rename(columns=RENAME)
