"""
Football Predictor — Streamlit Dashboard
Landing page and API status check.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dashboard.components.api_client import is_api_ready, get_models

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Football Predictor")
    st.caption("QMUL Final Year Project · Abdulaziz Alaskar")
    st.divider()

    ready = is_api_ready()
    if ready:
        st.success("API Connected", icon="✅")
    else:
        st.error("API Offline", icon="🔴")
        st.caption("Start the API:\n```\n.venv/bin/python -m uvicorn api.main:app --port 8000\n```")

    st.divider()
    st.caption("Navigate using the pages below:")

# ── Main content ──────────────────────────────────────────────────────────────
st.title("⚽ Football Predictor")
st.subheader("Three machine learning models for football analytics")
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏆 Match Outcome")
    st.markdown(
        "Predict **Home Win / Draw / Away Win** for a Premier League match. "
        "Each team's season-to-date stats are pulled automatically from the feature store."
    )
    st.metric("Test Accuracy", "52.85%", delta="+6.5pp vs dummy")
    st.caption("Model: XGBoost (Tuned, Chronological) · Cycle 1")

with col2:
    st.markdown("### 🎯 Expected Goals (xG)")
    st.markdown(
        "Predict the probability that a shot results in a goal. "
        "Click anywhere on the pitch to place your shot. Distance and angle are computed automatically."
    )
    st.metric("AUC-ROC", "0.8342", delta="+0.3342 vs dummy")
    st.caption("Model: XGBoost (Tuned, Chronological) · Cycle 2")

with col3:
    st.markdown("### 🏥 Injury Risk")
    st.markdown(
        "Predict whether a player is at high risk of missing 28+ days "
        "this season based on physical attributes and injury history."
    )
    st.metric("AUC-ROC", "0.6723", delta="Top of 0.60–0.70 literature range")
    st.caption("Model: XGBoost (Tuned, Chronological) · Cycle 3")

st.divider()

# ── Live model status ─────────────────────────────────────────────────────────
if ready:
    st.subheader("Loaded Models")
    try:
        models = get_models()
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            with col:
                primary = (
                    f"AUC {m['primary_value']:.4f}"
                    if m["primary_metric"] == "auc"
                    else f"{m['primary_value']*100:.2f}% accuracy"
                )
                st.markdown(f"**`{m['name']}`**")
                st.markdown(f"Cycle {m['cycle']} · {m['model_type']}")
                st.markdown(f"{primary} · {m['feature_count']} features")
    except Exception:
        st.warning("Could not load model metadata.")
else:
    st.info("Start the FastAPI server to see live model status.")

st.divider()
st.caption(
    "Football Predictor · QMUL BSc Computer Science & AI · "
    "Supervisor: Tayyab Ahmad Ansari · Student: 200612007"
)
