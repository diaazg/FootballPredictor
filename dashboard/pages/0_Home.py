from __future__ import annotations
"""
Football Analytics AI — Home Page
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dashboard.components.api_client import is_api_ready

# ── Main content ──────────────────────────────────────────────────────────────
st.title("Football Analytics AI: Predicting Matches, Goals and Injuries with Machine Learning")
st.subheader("Football analytics in three tools")
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏆 Match Outcome")
    st.markdown(
        "Predict **Home Win / Draw / Away Win** for a Premier League match. "
        "Each team's season-to-date stats are pulled automatically from the feature store."
    )

with col2:
    st.markdown("### 🎯 Expected Goals (xG)")
    st.markdown(
        "Predict the probability that a shot results in a goal. "
        "Click anywhere on the pitch to place your shot. Distance and angle are computed automatically."
    )

with col3:
    st.markdown("### 🏥 Injury Risk")
    st.markdown(
        "Predict whether a player is at high risk of missing 28+ days "
        "this season based on physical attributes and injury history."
    )

ready = is_api_ready()
if not ready:
    st.info("Start the FastAPI server to enable predictions.")

st.divider()
st.caption(
    "Football Analytics AI · QMUL BSc Computer Science & AI · "
    "Supervisor: Tayyab Ahmad Ansari · Student: 200612007"
)
