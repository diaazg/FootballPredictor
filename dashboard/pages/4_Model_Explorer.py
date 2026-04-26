"""
Page 4 — Model Explorer
Browse all evaluated models across all three cycles.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
from dashboard.components import api_client as api
from dashboard.components.charts import model_compare_table

st.set_page_config(page_title="Model Explorer", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("⚽ Football Predictor")
    if api.is_api_ready():
        st.success("API Connected", icon="✅")
    else:
        st.error("API Offline", icon="🔴")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Model Explorer")
st.write("Browse all models evaluated across the three prediction cycles.")
st.divider()

# ── Overview cards ────────────────────────────────────────────────────────────
try:
    models = api.get_models()
except Exception:
    st.error("Could not load model data. Is the API running?")
    st.stop()

cols = st.columns(3)
for col, m in zip(cols, models):
    with col:
        metric_label = "AUC-ROC" if m["primary_metric"] == "auc" else "Accuracy"
        metric_val   = f"{m['primary_value']:.4f}" if m["primary_metric"] == "auc" else f"{m['primary_value']*100:.2f}%"
        with st.container(border=True):
            st.markdown(f"### Cycle {m['cycle']} — `{m['name']}`")
            st.markdown(f"**{m['description']}**")
            st.write("")
            st.metric(metric_label, metric_val)
            st.caption(f"Algorithm: {m['model_type']}")
            st.caption(f"Features: {m['feature_count']}")
            st.caption(m["note"])

st.divider()

# ── Cycle-by-cycle comparison ─────────────────────────────────────────────────
cycle_labels = {
    "match":  "Cycle 1 — Match Outcome (W/D/L)",
    "xg":     "Cycle 2 — Expected Goals (xG)",
    "injury": "Cycle 3 — Player Injury Risk",
}

for model_name, label in cycle_labels.items():
    st.subheader(label)
    try:
        compare = api.get_model_compare(model_name)
        st.caption(f"Best model: **{compare['best']}**")
        st.plotly_chart(
            model_compare_table(compare["variants"]),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not load comparison for {model_name}: {e}")
    st.divider()

# ── Design decisions ──────────────────────────────────────────────────────────
st.subheader("Key Design Decisions")

with st.expander("Why AUC-ROC instead of Accuracy for Cycles 2 & 3?"):
    st.markdown("""
- **Cycle 2 (xG):** 89.2% of shots are not goals. A dummy model that always predicts "No Goal" achieves 89.2% accuracy but AUC=0.5 (random). AUC-ROC measures whether the model can actually rank goals above non-goals.
- **Cycle 3 (Injury):** 70.2% of players are in the High Injury class. A dummy achieves 70.1% accuracy but AUC=0.5. AUC forces the model to genuinely discriminate.
""")

with st.expander("Why XGBoost wins under chronological split (and why LR was the random-split winner)"):
    st.markdown("""
**Random-split result** (legacy): LR AUC 0.6220 beat XGBoost 0.6179. The conventional explanation
was that with only ~1,040 training rows, ensembles overfit to the CV folds.

**Chronological-split result** (deployed): XGBoost Tuned **AUC 0.6723** beat LR 0.6263.
The "small data → linear model wins" conclusion was an artefact of the random split itself —
when train and test came from the same time period, LR's smooth decision boundary
generalised better than a complex tree. Once the test set comes from later seasons,
XGBoost's ability to capture non-linear interactions (e.g. age × cumulative_days_injured)
wins out by ~5 AUC points.

This is a useful methodology lesson: how you split the data can change which model "wins".
""")

with st.expander("What is the 28-day injury threshold?"):
    st.markdown("""
The FinalYearProject used `season_days_injured > 0` as the target, which meant any player with even **one day** of injury was classified as injured — covering ~99.9% of players. This is a trivially imbalanced problem with no predictive value.

The 28-day threshold (4 weeks) is the widely-used sports medicine definition of a **significant injury**. It creates a meaningful 70/30 split and targets the injuries that actually affect a team's season.
""")

with st.expander("Why only England for Cycle 2 (xG)?"):
    st.markdown("""
FinalYearProject used all 5 European leagues (~33,000+ shots) but included **post-shot tags** (where in the goal/net the ball went) as features. These are not knowable before the shot — they're data leakage.

FootballPredictor uses **England only** (8,451 shots) with pre-shot features only: X, Y coordinates, foot used, match half, and player rank. Adding more leagues (France, Germany, Italy, Spain) would give ~50,000+ shots and could push AUC above 0.82 — identified as the highest-priority improvement.
""")
