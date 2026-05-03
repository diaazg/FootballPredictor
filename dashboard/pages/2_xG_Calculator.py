from __future__ import annotations
"""
Expected Goals (xG) Calculator (Cycle 2)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
from dashboard.components import api_client as api
from dashboard.components.charts import xg_pitch, shap_bar_chart

st.set_page_config(page_title="xG Calculator", page_icon="🎯", layout="wide")

# with st.sidebar:
#     st.title("⚽ Football Predictor")
#     if api.is_api_ready():
#         st.success("API Connected", icon="✅")
#     else:
#         st.error("API Offline", icon="🔴")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Expected Goals (xG) Calculator")
st.write(
    "Set the shot location and context below. **Distance and angle to goal are "
    "computed automatically** from the coordinates."
)
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
col_inputs, col_info = st.columns([1, 1])

with col_inputs:
    st.subheader("Shot Location")
    st.caption("X=100 is the goal line. Y=50 is the centre of the pitch width.")

    shot_x = st.slider("X — Distance along pitch (0=halfway, 100=goal line)", 50.0, 100.0, 85.0, 0.5)
    shot_y = st.slider("Y — Position across pitch (0=left, 100=right)", 0.0, 100.0, 50.0, 0.5)

    st.subheader("Shot Context")
    foot_choice = st.radio("Foot used", ["Right Foot", "Left Foot", "Header"], horizontal=True)
    left_foot  = 1 if foot_choice == "Left Foot"  else 0
    right_foot = 1 if foot_choice == "Right Foot" else 0
    header     = 1 if foot_choice == "Header"     else 0

    first_half  = 1 if st.radio("Match half", ["First Half", "Second Half"], horizontal=True) == "First Half" else 0
    player_rank = st.slider("Player rank score (playerank)", 0.0, 10.0, 6.5, 0.1)
    show_shap   = st.checkbox("Show SHAP explanation", value=False)

with col_info:
    st.subheader("Inputs used")
    st.markdown("""
| Input | Source |
|---|---|
| X, Y | Your sliders |
| Distance | Computed from X, Y → goal centre |
| Angle | Computed from X, Y → both posts |
| Foot type | Radio selection |
| Match half | Radio selection |
| Player rank | Slider |
""")

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Calculate xG", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        try:
            result = api.predict_xg(shot_x, shot_y, left_foot, right_foot, header, first_half, player_rank)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    xg          = result["xg"]
    prediction  = result["prediction"]
    distance_m  = result["distance_m"]
    angle_deg   = result["angle_deg"]
    confidence  = result["confidence"]

    # ── Result display ────────────────────────────────────────────────────────
    chart_col, metric_col = st.columns([3, 2])

    with chart_col:
        st.plotly_chart(xg_pitch(shot_x, shot_y, xg), use_container_width=True)

    with metric_col:
        st.subheader("Result")
        xg_colour = "#2ecc71" if xg >= 0.5 else ("#f39c12" if xg >= 0.2 else "#e74c3c")
        st.markdown(
            f"<h1 style='color:{xg_colour};font-size:64px;text-align:center;'>xG = {xg:.3f}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;font-size:18px;'>{prediction}</p>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Distance to goal",   f"{distance_m:.1f} m")
        st.metric("Angle to goal",      f"{angle_deg:.1f}°")
        st.metric("Confidence",         confidence.upper())
        st.write("")
        st.progress(xg, text=f"xG = {xg:.3f} (goal probability)")

        st.write("")
        if xg >= 0.5:
            st.success("High-quality chance (xG ≥ 0.50)")
        elif xg >= 0.2:
            st.warning("Moderate chance (0.20 ≤ xG < 0.50)")
        else:
            st.error("Low-quality chance (xG < 0.20)")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if show_shap:
        st.divider()
        st.subheader("SHAP Feature Explanation")
        with st.spinner("Computing SHAP values..."):
            try:
                shap_data = api.explain_xg(
                    shot_x, shot_y, left_foot, right_foot, header, first_half, player_rank
                )
                st.plotly_chart(
                    shap_bar_chart(shap_data["top_features"], "All Feature Impacts (SHAP)"),
                    use_container_width=True,
                )
                st.caption(f"Base value: {shap_data['base_value']:.4f}")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
