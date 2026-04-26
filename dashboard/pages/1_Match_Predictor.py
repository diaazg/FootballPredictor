"""
Page 1 — Match Outcome Predictor (Cycle 1)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import io
import pandas as pd
import streamlit as st
from dashboard.components import api_client as api
from dashboard.components.charts import match_probability_chart, shap_bar_chart

st.set_page_config(page_title="Match Predictor", page_icon="🏆", layout="wide")

# Stat display labels (rolling keys → human names)
_STAT_LABELS = {
    "avg_possession_5":      "Possession % (5-match avg)",
    "avg_shots_5":           "Shots (5-match avg)",
    "avg_shots_on_target_5": "Shots on Target (5-match avg)",
    "avg_pass_accuracy_5":   "Pass Accuracy % (5-match avg)",
    "avg_tackles_5":         "Tackles (5-match avg)",
    "avg_corners_5":         "Corners (5-match avg)",
    "avg_fouls_5":           "Fouls (5-match avg)",
    "avg_yellow_cards_5":    "Yellow Cards (5-match avg)",
}
STAT_KEYS = list(_STAT_LABELS.keys())

# ── Sidebar status ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Football Predictor")
    if api.is_api_ready():
        st.success("API Connected", icon="✅")
    else:
        st.error("API Offline", icon="🔴")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏆 Match Outcome Predictor")
st.caption("Cycle 1 · XGBoost (Tuned) · 57.33% accuracy · Premier League rolling stats")
st.divider()

# ── Load team list ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_teams():
    try:
        return api.get_teams()["teams"]
    except Exception:
        return []

teams = load_teams()
team_options = {
    t.get("name") or f"Team {t['team_id']}": t["team_id"]
    for t in teams
} if teams else {}

if not team_options:
    st.error("Could not load teams. Make sure the API is running.")
    st.stop()

# ── Team selection ────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Home Team")
    home_label = st.selectbox("Select Home Team", list(team_options.keys()), index=0, key="home")
    home_id    = team_options[home_label]

with col_right:
    st.subheader("Away Team")
    away_label = st.selectbox("Select Away Team", list(team_options.keys()), index=1, key="away")
    away_id    = team_options[away_label]

attendance = st.number_input(
    "Expected Attendance (0 = use dataset mean)",
    min_value=0, max_value=90000, value=40000, step=1000,
)

# ── Stats mode ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Rolling Statistics")
stats_mode = st.radio(
    "Stats source",
    ["Auto (load from feature store)", "Manual (enter stats)", "Upload CSV"],
    horizontal=True,
)

home_stats_override: dict | None = None
away_stats_override: dict | None = None

# ── Helper: load stored stats ─────────────────────────────────────────────────
def _fetch_stored(team_id: int) -> dict:
    try:
        return api.get_team(team_id)["stats"]
    except Exception:
        return {k: 0.0 for k in STAT_KEYS}

# ── AUTO mode ─────────────────────────────────────────────────────────────────
if stats_mode == "Auto (load from feature store)":
    col_h, col_a = st.columns(2)
    with col_h:
        with st.expander(f"{home_label} — latest rolling stats", expanded=False):
            try:
                home_stats = _fetch_stored(home_id)
                for k, label in _STAT_LABELS.items():
                    st.metric(label, f"{home_stats.get(k, 0):.2f}")
            except Exception:
                st.warning("Could not load home team stats.")
    with col_a:
        with st.expander(f"{away_label} — latest rolling stats", expanded=False):
            try:
                away_stats = _fetch_stored(away_id)
                for k, label in _STAT_LABELS.items():
                    st.metric(label, f"{away_stats.get(k, 0):.2f}")
            except Exception:
                st.warning("Could not load away team stats.")

# ── MANUAL mode ───────────────────────────────────────────────────────────────
elif stats_mode == "Manual (enter stats)":
    st.caption("Pre-filled with the latest stored values — edit any field before predicting.")

    stored_home = _fetch_stored(home_id)
    stored_away = _fetch_stored(away_id)

    col_h, col_a = st.columns(2)
    manual_home, manual_away = {}, {}

    with col_h:
        st.markdown(f"**{home_label}**")
        for k, label in _STAT_LABELS.items():
            manual_home[k] = st.number_input(
                label, value=float(stored_home.get(k, 0.0)),
                min_value=0.0, step=0.1, key=f"home_{k}",
            )

    with col_a:
        st.markdown(f"**{away_label}**")
        for k, label in _STAT_LABELS.items():
            manual_away[k] = st.number_input(
                label, value=float(stored_away.get(k, 0.0)),
                min_value=0.0, step=0.1, key=f"away_{k}",
            )

    home_stats_override = manual_home
    away_stats_override = manual_away

# ── CSV UPLOAD mode ───────────────────────────────────────────────────────────
else:
    csv_template = (
        "side," + ",".join(STAT_KEYS) + "\n"
        "home," + ",".join(["0.0"] * len(STAT_KEYS)) + "\n"
        "away," + ",".join(["0.0"] * len(STAT_KEYS)) + "\n"
    )
    st.download_button(
        "Download CSV template",
        data=csv_template,
        file_name="match_stats_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Fill in the `home` and `away` rows with 5-match rolling averages, "
        "then upload the file below."
    )

    uploaded = st.file_uploader("Upload stats CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            df_up.columns = df_up.columns.str.strip()
            df_up["side"] = df_up["side"].str.strip().str.lower()

            missing_cols = [k for k in STAT_KEYS if k not in df_up.columns]
            if missing_cols:
                st.error(f"CSV is missing columns: {missing_cols}")
            else:
                home_row = df_up[df_up["side"] == "home"]
                away_row = df_up[df_up["side"] == "away"]

                if home_row.empty or away_row.empty:
                    st.error("CSV must have a 'home' row and an 'away' row in the 'side' column.")
                else:
                    home_stats_override = {k: float(home_row.iloc[0][k]) for k in STAT_KEYS}
                    away_stats_override = {k: float(away_row.iloc[0][k]) for k in STAT_KEYS}

                    col_h, col_a = st.columns(2)
                    with col_h:
                        st.markdown(f"**{home_label}** (from CSV)")
                        for k, label in _STAT_LABELS.items():
                            st.metric(label, f"{home_stats_override[k]:.2f}")
                    with col_a:
                        st.markdown(f"**{away_label}** (from CSV)")
                        for k, label in _STAT_LABELS.items():
                            st.metric(label, f"{away_stats_override[k]:.2f}")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

show_shap = st.checkbox("Show SHAP feature explanation", value=False)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict Match Outcome", type="primary", use_container_width=True):
    if home_id == away_id:
        st.warning("Home and away team must be different.")
    elif stats_mode == "Upload CSV" and (home_stats_override is None or away_stats_override is None):
        st.warning("Please upload a valid CSV file before predicting.")
    else:
        with st.spinner("Running prediction..."):
            try:
                result = api.predict_match(
                    home_id, away_id, attendance,
                    home_stats=home_stats_override,
                    away_stats=away_stats_override,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        probs      = result["probabilities"]
        prediction = result["prediction"]
        confidence = result["confidence"]
        conf_colour = {"high": "green", "medium": "orange", "low": "red"}[confidence]

        st.subheader("Prediction")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col2:
            st.markdown(
                f"<h2 style='text-align:center;color:#1454a0;'>{prediction}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center;'>Confidence: "
                f"<span style='color:{conf_colour};font-weight:bold;'>{confidence.upper()}</span></p>",
                unsafe_allow_html=True,
            )

        st.plotly_chart(
            match_probability_chart(
                probs["home_win"], probs["draw"], probs["away_win"],
                home_label=home_label, away_label=away_label,
            ),
            use_container_width=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Home Win ({home_label})", f"{probs['home_win']*100:.1f}%")
        m2.metric("Draw",                     f"{probs['draw']*100:.1f}%")
        m3.metric(f"Away Win ({away_label})", f"{probs['away_win']*100:.1f}%")

        if show_shap:
            st.divider()
            st.subheader("SHAP Feature Explanation")
            st.caption("Shows which features pushed this prediction and by how much.")
            with st.spinner("Computing SHAP values..."):
                try:
                    shap_data = api.explain_match(
                        home_id, away_id, attendance, top_n=10,
                        home_stats=home_stats_override,
                        away_stats=away_stats_override,
                    )
                    st.plotly_chart(
                        shap_bar_chart(shap_data["top_features"], "Top 10 Feature Impacts"),
                        use_container_width=True,
                    )
                    st.caption(f"Base value (model prior): {shap_data['base_value']:.4f}")
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")
