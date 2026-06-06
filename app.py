"""
app.py — Premium AI House Price Predictor
Run: streamlit run app.py
"""
from __future__ import annotations

import io
import os
import time
import traceback
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Must be first Streamlit call
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import (
    PREMIUM_CSS,
    DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    META_PATH,
    align_features,
    df_to_csv_bytes,
    engineer_features,
    load_artifacts,
    load_dataset,
    load_meta,
    predict_with_interval,
    prepare_batch,
    prepare_single,
    save_meta,
    single_pred_pdf,
)

matplotlib.use("Agg")

# ── Inject CSS ─────────────────────────────────────────────────────────
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def section(icon: str, title: str):
    st.markdown(
        f"""<div class="section-header">
            <div class="icon">{icon}</div>
            <h3>{title}</h3>
        </div>""",
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "blue") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def fmt_price(v: float) -> str:
    return f"${v:,.0f}"


def _mpl_style():
    """Apply dark matplotlib style consistent with app theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#0a1220",
            "axes.facecolor": "#0a1220",
            "axes.edgecolor": "#1e3a5f",
            "axes.labelcolor": "#94a3b8",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "text.color": "#e2e8f0",
            "grid.color": "#1e3a5f",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "font.family": "sans-serif",
        }
    )


_mpl_style()

# Show tracebacks only when explicitly enabled for debugging.
DEBUG_MODE = os.getenv("APP_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def show_exception(prefix: str, ex: Exception):
    st.error(f"{prefix}: {ex}")
    if DEBUG_MODE:
        st.code(traceback.format_exc(), language="python")


def reliability_tag(z_score: float) -> tuple[str, str]:
    if z_score < 0.9:
        return "Low Risk", "green"
    if z_score < 1.6:
        return "Medium Risk", "amber"
    return "High Risk", "red"


def local_contributions(
    model,
    scaler,
    x_row: pd.DataFrame,
    feature_names: list[str],
    reference_values: dict[str, float],
) -> pd.DataFrame:
    """Approximate local feature contribution via one-feature-at-a-time replacement."""
    x_scaled = scaler.transform(x_row)
    pred_full = float(model.predict(x_scaled)[0])

    rows = []
    for f in feature_names:
        x_mut = x_row.copy()
        x_mut.loc[x_mut.index[0], f] = float(reference_values.get(f, 0.0))
        pred_mut = float(model.predict(scaler.transform(x_mut))[0])
        rows.append({"feature": f, "contribution": pred_full - pred_mut})

    out = pd.DataFrame(rows)
    out["abs_contribution"] = out["contribution"].abs()
    out = out.sort_values("abs_contribution", ascending=False)
    return out


if "prediction_logs" not in st.session_state:
    st.session_state.prediction_logs = []

# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """<div class="sidebar-logo">
            <div style="font-size:2.5rem">🏡</div>
            <h2>House Price AI</h2>
            <p>California Housing · ML Dashboard</p>
        </div>""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Overview", "Prediction", "Batch Predict", "Analytics", "Monitoring", "Data Explorer", "Retrain"],
        label_visibility="collapsed",
    )

    st.divider()

    # Real-time project stats
    meta = load_meta()
    if meta:
        st.markdown("**Project Stats**")
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:8px;margin-top:8px">
              <div class="metric-chip">Model <span class="val">{meta.get("model_type","—")}</span></div>
              <div class="metric-chip">R² <span class="val">{meta.get("r2","—")}</span></div>
              <div class="metric-chip">Trained <span class="val" style="font-size:11px">{meta.get("trained_at","—")[:10]}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Run `python train.py` to generate model artifacts.")

    try:
        df_side = load_dataset()
        rows, cols = df_side.shape
        st.markdown(
            f"""<div style="margin-top:16px">
              <div class="metric-chip">Rows <span class="val">{rows:,}</span></div>
              <div class="metric-chip">Cols <span class="val">{cols}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# Try loading artifacts globally (show error once)
# ══════════════════════════════════════════════════════════════════════
artifacts_ok = False
model = scaler = features = None
try:
    model, scaler, features = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    if page not in ("Retrain",):
        st.error(f"**Model artifacts missing.**\n\n{e}")


# ══════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## 🏡 AI House Price Prediction System")
    st.markdown(
        "<p style='color:#64748b;margin-top:-12px'>California Housing · Random Forest · Production-grade ML Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
                """
                <div style="margin-top:6px;margin-bottom:16px;max-width:980px;">
                    <div style="font-weight:600;color:#cbd5e1;margin-bottom:6px;">Project Description</div>
                    <p style="color:#94a3b8;line-height:1.7;margin:0;">
                        AI House Price Predictor is an end-to-end machine learning dashboard that estimates California house prices
                        from location and property features. It benchmarks multiple regression models, automatically keeps the best
                        performer, and supports single as well as batch predictions with interactive analytics.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
    )

    if meta and artifacts_ok:
        results = meta.get("results", {})
        best_name = meta.get("model_type", "")

        section("📊", "Model Comparison")
        rows_html = ""
        for mname, stats in results.items():
            row_class = "best-row" if mname == best_name else ""
            rows_html += f"""
            <tr class="{row_class}">
              <td>{mname}</td>
              <td class="mono">{stats["rmse"]:,.0f}</td>
              <td class="mono">{stats["mae"]:,.0f}</td>
              <td class="mono">{stats["r2"]}</td>
              <td class="mono">{stats["train_time"]}s</td>
            </tr>"""

        st.markdown(
            f"""<table class="styled-table">
              <thead><tr>
                <th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Train Time</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-top:12px'>{badge('★ Best model highlighted', 'blue')}</div>",
            unsafe_allow_html=True,
        )

    section("💡", "About This Project")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Pipeline summary**
            - Kaggle California Housing dataset (20K+ rows)
            - Feature engineering: rooms/household, bedroom ratio, occupancy density
            - One-hot encoding for ocean proximity
            - StandardScaler normalisation
            - 5 models benchmarked; best saved automatically
            """
        )
    with c2:
        st.markdown(
            """
            **Dashboard capabilities**
            - Single & batch price prediction
            - Confidence interval estimation
            - Feature importance explanations
            - Interactive analytics with filters
            - In-app model retraining
            - PDF & CSV export
            """
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE: Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown("## 💰 Predict House Price")

    if not artifacts_ok:
        st.stop()

    # Dataset medians for smart defaults
    try:
        df_raw = load_dataset()
        num_df = df_raw.select_dtypes(include="number").drop(
            columns=["Price"], errors="ignore"
        )
        medians = num_df.median().to_dict()
        mins = num_df.min().to_dict()
        maxs = num_df.max().to_dict()
    except Exception:
        medians = mins = maxs = {}

    # Raw input columns (before engineering)
    RAW_COLS = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
    ]
    OCEAN_OPTS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

    section("📐", "Property Details")
    col1, col2 = st.columns(2)
    inputs: dict = {}
    errors: list[str] = []

    left_fields = RAW_COLS[:4]
    right_fields = RAW_COLS[4:]

    def num_input(col, field):
        label = field.replace("_", " ").title()
        default = float(medians.get(field, 1.0))
        mn = float(mins.get(field, 0.0))
        mx = float(maxs.get(field, 1e9))
        v = col.number_input(
            label, value=default, min_value=mn, max_value=mx,
            help=f"Dataset range: {mn:.2f} – {mx:.2f}",
        )
        return v

    with col1:
        for f in left_fields:
            inputs[f] = num_input(col1, f)

    with col2:
        for f in right_fields:
            inputs[f] = num_input(col2, f)

    with col2:
        ocean = col2.selectbox("Ocean Proximity", OCEAN_OPTS, index=0)
        inputs["ocean_proximity"] = ocean

    # Validation
    if inputs["total_bedrooms"] > inputs["total_rooms"]:
        errors.append("Total bedrooms cannot exceed total rooms.")
    if inputs["households"] <= 0:
        errors.append("Households must be greater than 0.")

    st.divider()

    if st.button("🚀 Predict Price", use_container_width=True):
        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Running model inference…"):
                try:
                    X = prepare_single(inputs, features)
                    point, low, high = predict_with_interval(model, scaler, X)

                    # Reliability guardrail using distance from training distribution in z-space.
                    z_values = []
                    for c in RAW_COLS:
                        mu = float(medians.get(c, 0.0))
                        sigma = float(df_raw[c].std()) if "df_raw" in locals() and c in df_raw else 1.0
                        sigma = sigma if sigma > 1e-9 else 1.0
                        z_values.append(abs((float(inputs[c]) - mu) / sigma))
                    ood_score = float(np.mean(z_values)) if z_values else 0.0
                    risk_label, risk_kind = reliability_tag(ood_score)

                    st.markdown(
                        f"""<div class="pred-box">
                            <div style="font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">Estimated Market Value</div>
                            <div class="pred-price">{fmt_price(point)}</div>
                            <div class="pred-interval">90% confidence interval: <span>{fmt_price(low)}</span> – <span>{fmt_price(high)}</span></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"<div style='margin-top:10px'>{badge(f'Reliability: {risk_label} (score {ood_score:.2f})', risk_kind)}</div>",
                        unsafe_allow_html=True,
                    )
                    if risk_kind == "red":
                        st.warning(
                            "Input appears far from typical training data. Treat this prediction with caution."
                        )

                    st.session_state.prediction_logs.append(
                        {
                            "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
                            "type": "single",
                            "predicted_price": float(point),
                            "reliability": risk_label,
                            "ood_score": ood_score,
                            **{k: float(inputs[k]) for k in RAW_COLS},
                        }
                    )

                    # Per-prediction local explainability (modern XAI-style output).
                    section("🧠", "Prediction Explainability")
                    try:
                        df_ref = df_raw.drop(columns=["Price"], errors="ignore") if "df_raw" in locals() else pd.DataFrame()
                        x_ref = align_features(engineer_features(df_ref), features) if not df_ref.empty else X.copy()
                        ref_vals = x_ref.median(numeric_only=True).to_dict() if not x_ref.empty else {f: 0.0 for f in features}

                        contrib = local_contributions(model, scaler, X, features, ref_vals).head(8)
                        contrib["direction"] = np.where(contrib["contribution"] >= 0, "↑", "↓")
                        contrib["impact_$"] = contrib["contribution"].map(lambda v: f"{v:,.0f}")

                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in contrib["contribution"]]
                        y_labels = [f"{d} {f}" for d, f in zip(contrib["direction"], contrib["feature"])]
                        ax.barh(y_labels[::-1], contrib["contribution"].values[::-1], color=colors[::-1], alpha=0.9)
                        ax.axvline(0, color="#64748b", linewidth=1)
                        ax.set_xlabel("Estimated contribution to prediction ($)")
                        ax.set_title("Top Local Feature Contributions", color="#e2e8f0", fontsize=12)
                        ax.grid(True, axis="x", alpha=0.3)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                        st.dataframe(
                            contrib[["feature", "direction", "impact_$"]],
                            use_container_width=True,
                            hide_index=True,
                        )

                        top3 = [
                            (r["feature"], float(abs(r["contribution"])))
                            for _, r in contrib.head(3).iterrows()
                        ]
                    except Exception as explain_err:
                        st.info(f"Explainability unavailable for this model/input: {explain_err}")
                        top3 = []

                    # PDF download
                    section("📥", "Export Report")
                    try:
                        pdf_bytes = single_pred_pdf(
                            {k: v for k, v in inputs.items() if k != "ocean_proximity"},
                            point, low, high, top3,
                        )
                        st.download_button(
                            "📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name="house_price_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as pdf_err:
                        st.warning(f"PDF generation failed: {pdf_err}")

                except Exception as ex:
                    show_exception("Prediction failed", ex)


# ══════════════════════════════════════════════════════════════════════
# PAGE: Batch Predict
# ══════════════════════════════════════════════════════════════════════
elif page == "Batch Predict":
    st.markdown("## 📦 Batch Price Prediction")

    if not artifacts_ok:
        st.stop()

    section("📤", "Upload CSV")
    st.markdown(
        "<p style='color:#64748b;font-size:13px'>CSV must contain the same numeric columns as the training dataset (excluding the target). "
        "<code>ocean_proximity</code> is optional.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is None:
        st.markdown(
            """<div class="empty-state">
                <div class="es-icon">📂</div>
                <p>No file uploaded yet.<br>Upload a CSV to run batch predictions.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        try:
            df_batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # Schema validation
        REQUIRED = {
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
        }
        missing_cols = REQUIRED - set(df_batch.columns)
        if missing_cols:
            st.error(
                f"**Invalid schema.** Missing columns: `{', '.join(sorted(missing_cols))}`\n\n"
                "Please ensure your CSV has the same columns as the training data."
            )
            st.stop()

        st.success(f"✅ Valid schema detected — {len(df_batch):,} rows loaded.")

        section("👁 Preview", "Input Data")
        st.dataframe(df_batch.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner(f"Predicting for {len(df_batch):,} rows…"):
                try:
                    X_aligned = prepare_batch(df_batch.copy(), features)
                    X_scaled = scaler.transform(X_aligned)
                    preds = model.predict(X_scaled)

                    result_df = df_batch.copy()
                    result_df["Predicted_Price"] = preds.round(0).astype(int)

                    section("📊", "Prediction Results")
                    st.dataframe(
                        result_df[["longitude", "latitude", "housing_median_age",
                                   "median_income", "Predicted_Price"]].head(50),
                        use_container_width=True,
                    )

                    st.markdown(
                        f"""<div style="display:flex;gap:12px;margin-top:12px">
                          <div class="metric-chip">Mean <span class="val">{fmt_price(preds.mean())}</span></div>
                          <div class="metric-chip">Min <span class="val">{fmt_price(preds.min())}</span></div>
                          <div class="metric-chip">Max <span class="val">{fmt_price(preds.max())}</span></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    st.download_button(
                        "⬇ Download Results as CSV",
                        data=df_to_csv_bytes(result_df),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.session_state.prediction_logs.append(
                        {
                            "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
                            "type": "batch",
                            "batch_rows": int(len(result_df)),
                            "predicted_price": float(preds.mean()),
                            "reliability": "Batch",
                            "ood_score": float("nan"),
                        }
                    )
                except Exception as ex:
                    show_exception("Batch prediction failed", ex)


# ══════════════════════════════════════════════════════════════════════
# PAGE: Analytics
# ══════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown("## 📊 Analytics Dashboard")

    if not artifacts_ok:
        st.stop()

    try:
        df = load_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # ── Sidebar filters ───────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**Analytics Filters**")

        price_min = int(df["Price"].min()) if "Price" in df.columns else 0
        price_max = int(df["Price"].max()) if "Price" in df.columns else 1_000_000
        price_range = st.slider(
            "Price Range ($)",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            step=10_000,
        )

        ocean_opts = ["All"]
        if "ocean_proximity" in df.columns:
            ocean_opts += sorted(df["ocean_proximity"].dropna().unique().tolist())
        ocean_filter = st.selectbox("Ocean Proximity", ocean_opts)

        corr_thresh = st.slider(
            "Min |Correlation| with Price",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        )

    # Apply filters
    df_f = df.copy()
    if "Price" in df_f.columns:
        df_f = df_f[df_f["Price"].between(*price_range)]
    if ocean_filter != "All" and "ocean_proximity" in df_f.columns:
        df_f = df_f[df_f["ocean_proximity"] == ocean_filter]

    st.markdown(
        f"<div style='margin-bottom:16px'>{badge(f'{len(df_f):,} rows after filter', 'blue')}</div>",
        unsafe_allow_html=True,
    )

    tab_avp, tab_resid, tab_feat, tab_dist, tab_corr = st.tabs(
        ["Actual vs Predicted", "Residuals", "Feature Importance", "Distributions", "Correlation"]
    )

    # ── Prepare X/y for model evaluation ──────────────────────────────
    try:
        X_eval = df_f.drop("Price", axis=1, errors="ignore").copy()
        y_eval = df_f["Price"] if "Price" in df_f.columns else None

        X_eng = engineer_features(X_eval)
        X_aligned = align_features(X_eng, features)
        X_scaled = scaler.transform(X_aligned)
        preds_eval = model.predict(X_scaled)
    except Exception as ex:
        show_exception("Analytics preprocessing failed", ex)
        st.stop()

    # Actual vs Predicted
    with tab_avp:
        if y_eval is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_eval, preds_eval, alpha=0.25, color="#3b82f6", s=8, edgecolors="none")
            lims = [min(y_eval.min(), preds_eval.min()), max(y_eval.max(), preds_eval.max())]
            ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit", alpha=0.7)
            ax.set_xlabel("Actual Price ($)")
            ax.set_ylabel("Predicted Price ($)")
            ax.set_title("Actual vs Predicted", color="#e2e8f0", fontsize=13)
            ax.legend(framealpha=0.0, labelcolor="#94a3b8")
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Analytics CSV download
            out_df = pd.DataFrame({"actual": y_eval.values, "predicted": preds_eval})
            st.download_button(
                "⬇ Download Analytics CSV",
                data=df_to_csv_bytes(out_df),
                file_name="analytics_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("No target column available for this filter selection.")

    # Residuals
    with tab_resid:
        if y_eval is not None:
            residuals = y_eval.values - preds_eval
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(preds_eval, residuals, alpha=0.25, color="#8b5cf6", s=8, edgecolors="none")
            ax.axhline(0, color="#ef4444", linewidth=1.5, linestyle="--", alpha=0.8)
            ax.set_xlabel("Predicted Price ($)")
            ax.set_ylabel("Residual ($)")
            ax.set_title("Residual Analysis", color="#e2e8f0", fontsize=13)
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Feature Importance
    with tab_feat:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            feat_df = (
                pd.DataFrame({"feature": features, "importance": imp})
                .sort_values("importance", ascending=True)
            )
            max_imp = feat_df["importance"].max()
            bars_html = ""
            for _, row in feat_df.iterrows():
                pct = row["importance"] / max_imp * 100
                bars_html += f"""
                <div class="feat-bar-wrap">
                  <div class="feat-label">
                    <span>{row["feature"]}</span>
                    <span>{row["importance"]:.4f}</span>
                  </div>
                  <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width:{pct:.1f}%"></div>
                  </div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
        else:
            st.info("Feature importances not available for this model type.")

    # Distributions
    with tab_dist:
        num_cols = df_f.select_dtypes(include="number").columns.tolist()
        col_choice = st.selectbox("Select feature", num_cols, key="dist_col")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_f[col_choice].dropna(), bins=50, color="#3b82f6", edgecolor="none", alpha=0.8)
        ax.set_xlabel(col_choice)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col_choice}", color="#e2e8f0", fontsize=13)
        ax.grid(True, axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Correlation
    with tab_corr:
        num_df2 = df_f.select_dtypes(include="number")
        corr = num_df2.corr()

        if corr_thresh > 0 and "Price" in corr.columns:
            keep = corr["Price"].abs()[corr["Price"].abs() >= corr_thresh].index
            corr = corr.loc[keep, keep]

        fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
        import matplotlib.colors as mcolors
        cmap = matplotlib.colormaps.get_cmap("coolwarm")
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        ax.set_title("Correlation Heatmap", color="#e2e8f0", fontsize=13)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "#0f172a")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# PAGE: Monitoring
# ══════════════════════════════════════════════════════════════════════
elif page == "Monitoring":
    st.markdown("## 📈 Model Monitoring")

    logs = pd.DataFrame(st.session_state.prediction_logs)
    if logs.empty:
        st.info("No inference logs yet. Run single or batch predictions to populate monitoring.")
        st.stop()

    section("🩺", "Model Health Snapshot")
    total_preds = int(len(logs))
    single_logs = logs[logs["type"] == "single"].copy() if "type" in logs.columns else pd.DataFrame()
    latest_ts = str(logs.iloc[-1].get("ts", "—"))
    mean_pred = float(pd.to_numeric(logs["predicted_price"], errors="coerce").dropna().mean())

    st.markdown(
        f"""<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px">
          <div class="metric-chip">Predictions Logged <span class="val">{total_preds}</span></div>
          <div class="metric-chip">Average Prediction <span class="val">{fmt_price(mean_pred) if np.isfinite(mean_pred) else '—'}</span></div>
          <div class="metric-chip">Last Activity <span class="val" style="font-size:11px">{latest_ts}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not single_logs.empty and "reliability" in single_logs.columns:
        section("🛡", "Reliability Distribution")
        rel_counts = single_logs["reliability"].value_counts().rename_axis("risk").reset_index(name="count")
        st.dataframe(rel_counts, use_container_width=True, hide_index=True)

    section("🌊", "Input Drift vs Training Data")
    try:
        df_train = load_dataset()
        train_num = df_train.select_dtypes(include="number").drop(columns=["Price"], errors="ignore")
        raw_cols = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
        ]
        present_cols = [c for c in raw_cols if c in train_num.columns and c in single_logs.columns]

        if present_cols and not single_logs.empty:
            train_mean = train_num[present_cols].mean()
            train_std = train_num[present_cols].std().replace(0, np.nan)
            live_mean = single_logs[present_cols].apply(pd.to_numeric, errors="coerce").mean()
            drift = ((live_mean - train_mean).abs() / train_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            drift_df = pd.DataFrame(
                {
                    "feature": drift.index,
                    "normalized_shift_z": drift.values,
                }
            ).sort_values("normalized_shift_z", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(drift_df["feature"][::-1], drift_df["normalized_shift_z"][::-1], color="#f59e0b")
            ax.axvline(1.0, color="#ef4444", linestyle="--", linewidth=1.2, label="z=1 alert")
            ax.set_xlabel("Normalized shift (z-score units)")
            ax.set_title("Feature Drift (Live vs Training)", color="#e2e8f0", fontsize=12)
            ax.legend(framealpha=0.0, labelcolor="#94a3b8")
            ax.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.dataframe(drift_df.round(3), use_container_width=True, hide_index=True)

            max_drift = float(drift_df["normalized_shift_z"].max()) if not drift_df.empty else 0.0
            if max_drift >= 1.6:
                st.warning("High drift detected in recent inputs. Consider retraining with newer data.")
            elif max_drift >= 1.0:
                st.info("Moderate drift detected. Monitor closely as new predictions arrive.")
            else:
                st.success("Drift is currently low relative to training distribution.")
        else:
            st.info("Not enough single-prediction logs yet to compute drift.")
    except Exception as monitor_ex:
        show_exception("Monitoring failed", monitor_ex)

# ══════════════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("## 🔍 Data Explorer")

    try:
        df = load_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    tab_overview, tab_missing, tab_dtypes, tab_sample = st.tabs(
        ["Overview", "Missing Values", "Data Types", "Sample Rows"]
    )

    with tab_overview:
        st.markdown(
            f"""<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">
              <div class="metric-chip">Rows <span class="val">{df.shape[0]:,}</span></div>
              <div class="metric-chip">Columns <span class="val">{df.shape[1]}</span></div>
              <div class="metric-chip">Numeric cols <span class="val">{df.select_dtypes('number').shape[1]}</span></div>
              <div class="metric-chip">Categorical cols <span class="val">{df.select_dtypes('object').shape[1]}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab_missing:
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing = missing[missing["Missing Count"] > 0]
        if missing.empty:
            st.success("🎉 No missing values detected in the dataset.")
        else:
            st.dataframe(missing, use_container_width=True)

    with tab_dtypes:
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Data Type"]
        dtype_df["Sample Value"] = [str(df[c].iloc[0]) if len(df) > 0 else "" for c in df.columns]
        st.dataframe(dtype_df, use_container_width=True)

    with tab_sample:
        n = st.slider("Rows to display", 5, 100, 20)
        st.dataframe(df.head(n), use_container_width=True)
        st.download_button(
            "⬇ Download Dataset Sample",
            data=df_to_csv_bytes(df.head(500)),
            file_name="dataset_sample.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE: Retrain
# ══════════════════════════════════════════════════════════════════════
elif page == "Retrain":
    st.markdown("## 🔄 Model Retraining")

    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Place `housing.csv` inside the `data/` folder."
        )
        st.stop()

    st.markdown(
        "<p style='color:#64748b'>Retrain all candidate models on the current dataset and save the best one. "
        "This will overwrite <code>model/model.pkl</code>, <code>scaler.pkl</code>, and <code>features.pkl</code>.</p>",
        unsafe_allow_html=True,
    )

    existing_meta = load_meta()
    if existing_meta:
        st.markdown(
            f"**Currently deployed:** {existing_meta.get('model_type','—')} "
            f"(R² = {existing_meta.get('r2','—')}, trained {existing_meta.get('trained_at','—')[:10]})"
        )

    st.divider()

    if st.button("🚀 Start Retraining", use_container_width=True):
        progress = st.progress(0, text="Initialising…")
        status_box = st.empty()
        log_box = st.empty()
        logs: list[str] = []

        try:
            from preprocess import load_and_preprocess
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
            import time as _time

            status_box.info("⏳ Loading and preprocessing data…")
            progress.progress(5, text="Loading data…")
            (X_train, X_test, y_train, y_test), new_scaler, new_features = load_and_preprocess(
                str(DATA_PATH)
            )
            logs.append(f"✅ Data loaded — {len(X_train) + len(X_test):,} rows")

            candidates = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            }

            results: dict = {}
            best_model = None
            best_r2 = float("-inf")
            best_name = ""

            total = len(candidates)
            for idx, (name, m) in enumerate(candidates.items()):
                pct = 10 + int((idx / total) * 80)
                progress.progress(pct, text=f"Training {name}…")
                status_box.info(f"⏳ Training **{name}**…")

                t0 = _time.perf_counter()
                m.fit(X_train, y_train)
                elapsed = _time.perf_counter() - t0

                p = m.predict(X_test)
                r2 = float(r2_score(y_test, p))
                rmse = float(root_mean_squared_error(y_test, p))
                mae = float(mean_absolute_error(y_test, p))

                results[name] = {"r2": round(r2, 4), "rmse": round(rmse, 2),
                                 "mae": round(mae, 2), "train_time": round(elapsed, 3)}
                logs.append(f"✅ {name}: R²={r2:.4f}  RMSE={rmse:,.0f}  ({elapsed:.2f}s)")
                log_box.code("\n".join(logs))

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = m
                    best_name = name

            progress.progress(92, text="Saving artifacts…")
            status_box.info("⏳ Saving model artifacts…")

            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(best_model, MODEL_PATH)
            joblib.dump(new_scaler, SCALER_PATH)
            joblib.dump(new_features, FEATURES_PATH)
            save_meta(best_name, best_r2)

            # Write full meta with results
            import json, datetime
            full_meta = {
                "model_type": best_name,
                "r2": round(best_r2, 4),
                "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "results": results,
            }
            META_PATH.write_text(json.dumps(full_meta, indent=2))

            # Invalidate cache so sidebar refreshes
            load_artifacts.clear()

            progress.progress(100, text="Done!")
            status_box.success(
                f"✅ Retraining complete! Best model: **{best_name}** (R² = {best_r2:.4f})"
            )
            logs.append(f"\n🏆 Best: {best_name}  R²={best_r2:.4f}")
            log_box.code("\n".join(logs))

        except Exception as ex:
            progress.progress(0, text="Failed")
            show_exception("Retraining failed", ex)


# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:12px;color:#334155'>"
    "⚡ AI House Price Predictor · PAF-IAST · Built with Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
