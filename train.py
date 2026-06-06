"""
train.py — trains multiple regressors, saves best model + metadata.
Run: python train.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from preprocess import load_and_preprocess

DATA_PATH = Path("data") / "housing.csv"
MODEL_DIR = Path("model")


def train(data_path: str = str(DATA_PATH)) -> dict:
    """Train all models; return results dict keyed by model name."""
    (X_train, X_test, y_train, y_test), scaler, feature_names = load_and_preprocess(
        data_path
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results: dict[str, dict] = {}
    best_model = None
    best_r2 = float("-inf")

    print("\n── Model Performance ──────────────────────────────────────\n")

    for name, model in candidates.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        preds = model.predict(X_test)
        rmse = float(root_mean_squared_error(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        results[name] = {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "train_time": round(elapsed, 3),
        }

        print(f"{name}")
        print(f"  RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  R²: {r2:.4f}  Time: {elapsed:.3f}s")
        print("  " + "─" * 50)

        if r2 > best_r2:
            best_r2 = r2
            best_model = (name, model)

    # ── Persist artifacts ──────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    best_name, best_estimator = best_model  # type: ignore[misc]

    joblib.dump(best_estimator, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODEL_DIR / "features.pkl")

    meta = {
        "model_type": best_name,
        "r2": round(best_r2, 4),
        "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✅  Best model: {best_name}  (R² = {best_r2:.4f})")
    print("✅  Artifacts saved to model/")
    return meta


if __name__ == "__main__":
    train()
