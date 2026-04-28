"""
Page 3 — Player Injury Risk Predictor (Cycle 3)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
from dashboard.components import api_client as api
from dashboard.components.charts import injury_gauge, shap_bar_chart

st.set_page_config(page_title="Injury Risk", page_icon="🏥", layout="wide")

with st.sidebar:
    st.title("⚽ Football Predictor")
    if api.is_api_ready():
        st.success("API Connected", icon="✅")
    else:
        st.error("API Offline", icon="🔴")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Player Injury Risk Predictor")
st.write(
    "Enter a player's physical attributes and injury history to predict "
    "whether they are at high risk of missing **28+ days** this season."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Player Profile")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Physical Attributes**")
    height_cm   = st.number_input("Height (cm)",      min_value=155.0, max_value=210.0, value=181.0, step=1.0)
    weight_kg   = st.number_input("Weight (kg)",      min_value=55.0,  max_value=110.0, value=76.0,  step=1.0)
    bmi         = st.number_input("BMI",              min_value=16.0,  max_value=35.0,  value=round(weight_kg / (height_cm/100)**2, 1), step=0.1)
    st.markdown("**FIFA Attributes**")
    pace        = st.slider("Pace (FIFA)",     0, 100, 72)
    physic      = st.slider("Physic (FIFA)",   0, 100, 75)
    fifa_rating = st.slider("Overall Rating",  50, 99, 78)

with col2:
    st.markdown("**Demographics**")
    age = st.number_input("Age", min_value=16, max_value=40, value=26, step=1)
    position_map = {
        "Goalkeeper (1)": 1, "Defender (2)": 2, "Midfielder (3)": 3, "Attacker (4)": 4
    }
    pos_label       = st.selectbox("Position", list(position_map.keys()), index=2)
    position_numeric = position_map[pos_label]

    work_rate_map = {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3}
    wr_label         = st.selectbox("Work Rate", list(work_rate_map.keys()), index=1)
    work_rate_numeric = work_rate_map[wr_label]

    st.markdown("**Career Totals (before this season)**")
    cumulative_minutes_played = st.number_input("Career minutes played", 0, 60000, 8200, 100)
    cumulative_games_played   = st.number_input("Career games played",   0, 700,   95,   1)
    cumulative_days_injured   = st.number_input("Career days injured",   0, 1000,  45,   1)

with col3:
    st.markdown("**Previous Season Stats**")
    minutes_per_game_prev_seasons      = st.number_input("Avg min/game (prev seasons)",    0.0, 90.0,  72.0, 1.0)
    avg_games_per_season_prev_seasons  = st.number_input("Avg games/season (prev seasons)", 0.0, 60.0,  28.0, 1.0)
    avg_days_injured_prev_seasons      = st.number_input("Avg injury days/season (prev)",   0.0, 200.0,  8.5, 0.5)
    season_days_injured_prev_season    = st.number_input("Injury days last season",          0.0, 365.0, 12.0, 1.0)
    significant_injury_prev_season     = 1 if st.radio(
        "Missed 28+ days last season?", ["No", "Yes"], horizontal=True
    ) == "Yes" else 0

show_shap = st.checkbox("Show SHAP feature explanation", value=False)
st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Injury Risk", type="primary", use_container_width=True):
    features = {
        "height_cm":                         float(height_cm),
        "weight_kg":                         float(weight_kg),
        "pace":                              float(pace),
        "physic":                            float(physic),
        "fifa_rating":                       float(fifa_rating),
        "age":                               float(age),
        "bmi":                               float(bmi),
        "work_rate_numeric":                 float(work_rate_numeric),
        "position_numeric":                  float(position_numeric),
        "cumulative_minutes_played":         float(cumulative_minutes_played),
        "cumulative_games_played":           float(cumulative_games_played),
        "cumulative_days_injured":           float(cumulative_days_injured),
        "minutes_per_game_prev_seasons":     float(minutes_per_game_prev_seasons),
        "avg_days_injured_prev_seasons":     float(avg_days_injured_prev_seasons),
        "avg_games_per_season_prev_seasons": float(avg_games_per_season_prev_seasons),
        "significant_injury_prev_season":    int(significant_injury_prev_season),
        "season_days_injured_prev_season":   float(season_days_injured_prev_season),
    }

    with st.spinner("Assessing injury risk..."):
        try:
            result = api.predict_injury(features)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    prob_high  = result["probabilities"]["high_injury"]
    prob_low   = result["probabilities"]["low_injury"]
    prediction = result["prediction"]
    confidence = result["confidence"]

    # ── Result display ────────────────────────────────────────────────────────
    gauge_col, detail_col = st.columns([1, 1])

    with gauge_col:
        st.plotly_chart(injury_gauge(prob_high), use_container_width=True)

    with detail_col:
        st.subheader("Risk Assessment")
        if prob_high >= 0.6:
            st.error(f"**{prediction}**")
        elif prob_high >= 0.4:
            st.warning(f"**{prediction}**")
        else:
            st.success(f"**{prediction}**")

        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("High Injury Probability", f"{prob_high*100:.1f}%")
        m2.metric("Low Injury Probability",  f"{prob_low*100:.1f}%")
        st.metric("Confidence", confidence.upper())

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if show_shap:
        st.divider()
        st.subheader("SHAP Feature Explanation")
        st.caption("Which features most influenced this injury risk prediction.")
        with st.spinner("Computing SHAP values..."):
            try:
                shap_data = api.explain_injury(features, top_n=10)
                st.plotly_chart(
                    shap_bar_chart(shap_data["top_features"], "Top 10 Feature Impacts (SHAP)"),
                    use_container_width=True,
                )
                st.caption(f"Base value (model prior): {shap_data['base_value']:.4f}")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
