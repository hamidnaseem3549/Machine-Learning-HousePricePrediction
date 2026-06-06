"""Basic smoke checks for preprocessing, artifacts, and prediction."""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess import load_and_preprocess
from train import DATA_PATH, MODEL_DIR, train
from utils import FEATURES_PATH, MODEL_PATH, SCALER_PATH, prepare_single


def ensure_artifacts() -> None:
    required = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
    if all(p.exists() for p in required):
        return
    print("Artifacts missing. Running training once...")
    train(str(DATA_PATH))


def test_preprocess() -> None:
    (x_train, x_test, y_train, y_test), scaler, feature_names = load_and_preprocess(str(DATA_PATH))
    assert len(x_train) > 0 and len(x_test) > 0, "Empty train/test split"
    assert len(y_train) > 0 and len(y_test) > 0, "Empty train/test targets"
    assert hasattr(scaler, "mean_"), "Scaler not fitted"
    assert len(feature_names) > 0, "No features produced"


def test_prediction_smoke() -> None:
    ensure_artifacts()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = list(joblib.load(FEATURES_PATH))

    df = pd.read_csv(DATA_PATH)
    row = df.iloc[0].to_dict()

    # Remove target columns if present and keep one categorical feature if available.
    row.pop("Price", None)
    row.pop("median_house_value", None)
    if "ocean_proximity" not in row:
        row["ocean_proximity"] = "INLAND"

    x = prepare_single(row, features)
    x_scaled = scaler.transform(x)
    pred = float(model.predict(x_scaled)[0])

    assert pred >= 0, "Prediction should be non-negative"


def main() -> None:
    assert (PROJECT_ROOT / "data" / "housing.csv").exists(), "Dataset file missing"

    test_preprocess()
    test_prediction_smoke()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
