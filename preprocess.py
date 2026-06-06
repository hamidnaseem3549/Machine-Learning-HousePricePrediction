"""
preprocess.py — data loading and preprocessing pipeline.
Strict consistency with analytics and app.py feature alignment.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(path: str):
    """
    Returns:
        (X_train, X_test, y_train, y_test), scaler, feature_names (Index)
    """
    df = pd.read_csv(path)

    # ── Target column normalisation ────────────────────────────────────
    if "median_house_value" in df.columns:
        df = df.rename(columns={"median_house_value": "Price"})

    # Strict target requirement for reliable training/evaluation.
    if "Price" not in df.columns:
        raise ValueError(
            "Target column missing. Expected 'Price' or 'median_house_value'."
        )

    # ── Missing values ─────────────────────────────────────────────────
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)

    # ── Feature engineering (Kaggle schema) ───────────────────────────
    kaggle_cols = {"total_rooms", "households", "total_bedrooms", "population"}
    if kaggle_cols.issubset(df.columns):
        df["Rooms_per_Household"] = df["total_rooms"] / df["households"]
        df["Bedrooms_per_Room"] = df["total_bedrooms"] / df["total_rooms"]
        df["Population_per_Household"] = df["population"] / df["households"]

    # ── Categorical encoding ───────────────────────────────────────────
    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    # Split first, then fit transformations on train only to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train, y_test), scaler, X.columns
