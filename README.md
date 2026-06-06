# AI House Price Predictor

Semester project for California housing price prediction using scikit-learn and a Streamlit dashboard.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Train models and generate artifacts:
   - `python train.py`
4. Run app:
   - `streamlit run app.py`

## Project Assumptions

- Input data follows the California housing schema used in `data/housing.csv`.
- Target column must exist as either `Price` or `median_house_value`.
- Inference inputs use the same semantic feature definitions as training data.
- Categorical handling is limited to `ocean_proximity` one-hot encoding.
- Confidence intervals are approximate and model-dependent.

## Limitations

- Current model quality and generalization are tied to this dataset only.
- No online/continuous learning pipeline is implemented.
- No production authentication or role-based access control is included.
- Batch input validation checks required columns but does not enforce all domain constraints.
- Drift detection and automated retraining triggers are not implemented.

## Reproducibility

- Dependency versions are pinned in `requirements.txt` for deterministic setup.
- Training and split randomness use a fixed seed (`random_state=42`) where configured.

## Smoke Test

Run a quick integrity check before demos/submission:

- `python tests/smoke_test.py`

This test validates preprocessing, artifact availability/training, and one prediction pass.
