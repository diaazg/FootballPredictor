from __future__ import annotations
"""
Football Analytics AI — Interactive Analysis and Prediction Platform
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dashboard.components.api_client import is_api_ready

st.set_page_config(
    page_title="Football Analytics AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation (hidden so we can manually place links below our sidebar content)
home_page    = st.Page("pages/0_Home.py",              title="Home",           icon="🏠", default=True)
match_page   = st.Page("pages/1_Match_Predictor.py",  title="Match Outcome",  icon="🏆")
xg_page      = st.Page("pages/2_xG_Calculator.py",    title="xG Calculator",  icon="🎯")
injury_page  = st.Page("pages/3_Injury_Risk.py",       title="Injury Risk",    icon="🏥")

pg = st.navigation([home_page, match_page, xg_page, injury_page], position="hidden")

# ── Sidebar (full manual control of order) ────────────────────────────────────
with st.sidebar:
    st.title("⚽ Football Analytics AI")
    st.caption("QMUL Final Year Project · Abdulaziz Alaskar")

    ready = is_api_ready()
    if ready:
        st.success("API Connected", icon="✅")
    else:
        st.error("API Offline", icon="🔴")
        st.caption("Start the API:\n```\n.venv/bin/python -m uvicorn api.main:app --port 8000\n```")

    st.divider()

    # ── Page links (manually placed BELOW title & status) ────────────────────
    st.sidebar.page_link("pages/0_Home.py",              label="Home",           icon="🏠")
    st.sidebar.page_link("pages/1_Match_Predictor.py",  label="Match Outcome",  icon="🏆")
    st.sidebar.page_link("pages/2_xG_Calculator.py",    label="xG Calculator",  icon="🎯")
    st.sidebar.page_link("pages/3_Injury_Risk.py",       label="Injury Risk",    icon="🏥")

pg.run()
