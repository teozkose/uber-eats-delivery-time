"""Load the raw delivery records and turn them into a modeling table.

Output: data/features.parquet with one row per delivery, the target `time_taken_min`,
numeric features, and one-hot encoded categoricals.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "cleaned_data.csv")
OUT = os.path.join(ROOT, "data", "features.parquet")

TARGET = "time_taken_min"
NUMERIC = ["courier_age", "courier_rating", "distance_km", "vehicle_condition",
           "multiple_deliveries", "pickup_delay_min", "order_hour", "is_weekend", "is_peak_hour"]
CATEGORICAL = ["weather", "traffic", "order_type", "vehicle_type", "festival", "city"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two arrays of coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_raw(path: str = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Delivery_person_Age": "courier_age",
        "Delivery_person_Ratings": "courier_rating",
        "Weatherconditions": "weather",
        "Road_traffic_density": "traffic",
        "Vehicle_condition": "vehicle_condition",
        "Type_of_order": "order_type",
        "Type_of_vehicle": "vehicle_type",
        "Festival": "festival",
        "City": "city",
        "Time_taken(min)": TARGET,
    })
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["courier_age", "courier_rating", "vehicle_condition", "multiple_deliveries", TARGET,
              "Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Some coordinates are stored as negatives or zeros; use absolute values and drop zero rows.
    for c in ["Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]:
        df[c] = df[c].abs()
    df = df[(df["Restaurant_latitude"] > 1) & (df["Delivery_location_latitude"] > 1)]

    df["distance_km"] = haversine_km(df["Restaurant_latitude"], df["Restaurant_longitude"],
                                     df["Delivery_location_latitude"], df["Delivery_location_longitude"])
    # A handful of coordinate pairs are hundreds of km apart, which is not a food delivery.
    df = df[df["distance_km"] <= 50]

    ordered = pd.to_datetime(df["Time_Orderd"], errors="coerce")
    picked = pd.to_datetime(df["Time_Order_picked"], errors="coerce")
    delay = (picked - ordered).dt.total_seconds() / 60
    # Orders picked up after midnight show up as large negative gaps; wrap them around the day.
    delay = delay.where(delay >= 0, delay + 24 * 60)
    df["pickup_delay_min"] = delay
    df["order_hour"] = ordered.dt.hour
    df["is_weekend"] = ordered.dt.dayofweek.isin([5, 6]).astype(float)
    df["is_peak_hour"] = ordered.dt.hour.isin([12, 13, 19, 20, 21]).astype(float)

    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.replace("conditions ", "", regex=False)

    df = df.dropna(subset=[TARGET, "order_hour", "pickup_delay_min"])
    df = df[(df["pickup_delay_min"] >= 0) & (df["pickup_delay_min"] <= 60)]

    table = df[[TARGET] + NUMERIC + CATEGORICAL].copy()
    table = pd.get_dummies(table, columns=CATEGORICAL, drop_first=False, dtype=float)
    for c in NUMERIC:
        table[c] = table[c].fillna(table[c].median())
    return table.reset_index(drop=True)


def feature_columns(table: pd.DataFrame) -> list:
    return [c for c in table.columns if c != TARGET]


if __name__ == "__main__":
    raw = load_raw()
    table = build_features(raw)
    table.to_parquet(OUT, index=False)
    print(f"raw rows: {len(raw):,} -> modeling rows: {len(table):,}, features: {len(feature_columns(table))}")
    print(f"saved {OUT}")
