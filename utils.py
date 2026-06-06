"""
utils.py — shared helpers for the House Price Predictor app.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────── paths ────────────────────────────────
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"
DATA_PATH = Path("data") / "housing.csv"
META_PATH = MODEL_DIR / "meta.json"


# ─────────────────────────────── loaders ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler, and feature list. Raises FileNotFoundError with guidance."""
    missing = [p for p in (MODEL_PATH, SCALER_PATH, FEATURES_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required model artifacts not found: {[str(m) for m in missing]}.\n"
            "Run `python train.py` first to generate them."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = list(joblib.load(FEATURES_PATH))
    return model, scaler, features


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    df = pd.read_csv(DATA_PATH)
    if "median_house_value" in df.columns:
        df = df.rename(columns={"median_house_value": "Price"})
    return df


# ─────────────────────────────── preprocessing ────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror preprocess.py feature engineering (Kaggle schema)."""
    df = df.copy()
    kaggle_cols = {"total_rooms", "households", "total_bedrooms", "population"}
    if kaggle_cols.issubset(df.columns):
        df["Rooms_per_Household"] = df["total_rooms"] / df["households"]
        df["Bedrooms_per_Room"] = df["total_bedrooms"] / df["total_rooms"]
        df["Population_per_Household"] = df["population"] / df["households"]
    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
    return df


def align_features(X: pd.DataFrame, features: list) -> pd.DataFrame:
    """Reindex to saved feature list, filling missing one-hot cols with 0."""
    return X.reindex(columns=features, fill_value=0.0)


def prepare_single(row: dict, features: list) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df = engineer_features(df)
    return align_features(df, features)


def prepare_batch(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df = engineer_features(df)
    return align_features(df, features)


# ─────────────────────────────── prediction helpers ───────────────────
def predict_with_interval(
    model, scaler, X_aligned: pd.DataFrame, confidence: float = 0.90
) -> Tuple[float, float, float]:
    """
    Returns (point_estimate, lower_bound, upper_bound).
    Uses tree variance for RF/GB; falls back to ±10 % for linear models.
    """
    X_scaled = scaler.transform(X_aligned)
    point = model.predict(X_scaled)[0]

    try:
        preds_tree = np.array([t.predict(X_scaled)[0] for t in model.estimators_])
        std = preds_tree.std()
        z = 1.645 if confidence == 0.90 else 1.96
        return point, max(0, point - z * std), point + z * std
    except AttributeError:
        margin = point * 0.10
        return point, max(0, point - margin), point + margin


# ─────────────────────────────── dataset stats ────────────────────────
def dataset_stats(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    return {"rows": rows, "cols": cols}


# ─────────────────────────────── download helpers ─────────────────────
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def single_pred_pdf(inputs: dict, price: float, low: float, high: float,
                    top_features: list) -> bytes:
    """Generate a simple PDF report without ReportLab's complex layout engine."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    import datetime

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("House Price Prediction Report", styles["Title"]))
    elems.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("Predicted Price", styles["Heading2"]))
    price_data = [
        ["Estimate", f"${price:,.0f}"],
        ["Lower bound", f"${low:,.0f}"],
        ["Upper bound", f"${high:,.0f}"],
    ]
    t = Table(price_data, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("Top Feature Contributors", styles["Heading2"]))
    for fname, imp in top_features:
        elems.append(Paragraph(f"• {fname}: {imp:.4f}", styles["Normal"]))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("Input Values", styles["Heading2"]))
    input_rows = [["Feature", "Value"]] + [[k, str(round(v, 4))] for k, v in inputs.items()]
    t2 = Table(input_rows, colWidths=[250, 150])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elems.append(t2)

    doc.build(elems)
    return buf.getvalue()


# ─────────────────────────────── meta ─────────────────────────────────
def save_meta(model_type: str, r2: float):
    import json, datetime
    MODEL_DIR.mkdir(exist_ok=True)
    meta = {
        "model_type": model_type,
        "r2": round(r2, 4),
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    META_PATH.write_text(json.dumps(meta))


def load_meta() -> dict:
    import json
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


# ─────────────────────────────── CSS ──────────────────────────────────
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #0d1421 !important;
    border-right: 1px solid #1e2d45 !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #0f1a2e 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: border-color 0.2s ease;
}
.card:hover { border-color: #2d6aad; }

/* ── Metric chips ── */
.metric-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0f1e35;
    border: 1px solid #1e3a5f;
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    color: #7eb8f7;
    margin: 4px;
}
.metric-chip .val {
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #0a2040, #0d1f3c);
    border: 1px solid #2563eb;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(37,99,235,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.pred-price {
    font-size: 3rem;
    font-weight: 700;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
}
.pred-interval {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 8px;
}
.pred-interval span { color: #94a3b8; font-weight: 500; }

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #1e3a5f;
}
.section-header .icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.section-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.01em;
}

/* ── Table ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: 'Space Grotesk', sans-serif;
}
.styled-table th {
    background: #0f1e35;
    color: #7eb8f7;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #1e3a5f;
}
.styled-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #121e30;
    color: #cbd5e1;
}
.styled-table tr:hover td { background: #0f1e35; }
.styled-table .best-row td { color: #60a5fa; font-weight: 600; }
.styled-table .best-row td:first-child::before { content: "★ "; }

/* ── Feature importance bar ── */
.feat-bar-wrap { margin: 6px 0; }
.feat-label {
    display: flex; justify-content: space-between;
    font-size: 12px; color: #94a3b8; margin-bottom: 4px;
}
.feat-bar-bg {
    background: #0f1e35;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.feat-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #1d4ed8, #60a5fa);
    transition: width 0.6s ease;
}

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-green { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-blue  { background: #0c1a3d; color: #60a5fa; border: 1px solid #1d4ed8; }
.badge-amber { background: #1c1100; color: #fbbf24; border: 1px solid #92400e; }
.badge-red   { background: #1c0505; color: #f87171; border: 1px solid #7f1d1d; }

/* ── Sidebar nav ── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    color: #94a3b8 !important;
    padding: 6px 0 !important;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    height: 2.8em !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: #2563eb !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #0d1421 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid #1e3a5f !important;
}
[data-baseweb="tab"] {
    color: #64748b !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1d4ed8 !important;
    color: white !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] > div > div {
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: #1d4ed8 !important;
}

/* ── Alert/toast tweaks ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Keep Streamlit controls visible (menu + sidebar toggle) ── */
footer { visibility: hidden; }

/* ── Divider ── */
hr { border-color: #1e3a5f !important; margin: 24px 0 !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #475569;
}
.empty-state .es-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state p { font-size: 14px; line-height: 1.6; }

/* ── Mono values ── */
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    padding: 0 0 24px;
    text-align: center;
}
.sidebar-logo h2 {
    font-size: 1.1rem !important;
    color: #60a5fa !important;
    margin: 8px 0 2px !important;
}
.sidebar-logo p {
    font-size: 11px;
    color: #475569;
}
</style>
"""
