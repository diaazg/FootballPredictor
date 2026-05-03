# Smart Football Analytics: Predicting Matches, Goals, and Injuries with Machine Learning

**QMUL BSc Computer Science & Artificial Intelligence — Final Year Project**  
**Student:** Abdulaziz Alaskar (200612007)  
**Supervisor:** Tayyab Ahmad Ansari  
**Academic Year:** 2025–2026

---

## Project Summary

This project builds a unified football analytics system that combines three independent machine learning models into a single deployable application:

| Cycle | Task | Algorithm | Key Metric |
|-------|------|-----------|------------|
| 1 | Match Outcome (Win/Draw/Loss) | XGBoost (tuned, chronological) | 52.85% accuracy |
| 2 | Expected Goals (xG) per shot | XGBoost (tuned, random) | AUC-ROC 0.8183 |
| 3 | Player Injury Risk (28+ days) | LightGBM (tuned, chronological) | AUC-ROC 0.6800 |

All three models are served through a single **FastAPI** backend and visualised in a **Streamlit** dashboard. A "pipeline" mode chains all three models together: shot data feeds the xG model, player data feeds the injury model, and both outputs adjust the final match probability.

---

## Repository Structure

```
final_year_project/
│
├── config.py                    # Shared path configuration (Paths class)
├── requirements.txt             # All Python dependencies
│
├── data/
│   ├── raw/                     # Original unmodified source datasets
│   │   ├── premier_league_matches.csv     # PL matches 2000–2018 (6,840 rows)
│   │   ├── events_England.json            # Wyscout shot events (England 2017/18)
│   │   ├── playerank.json                 # Wyscout player quality scores
│   │   └── player_injuries.csv            # Player injury records (~1,300 player-seasons)
│   └── processed/               # Feature-engineered CSVs (written by preprocessing notebooks)
│       ├── premier_league_matches_processed.csv
│       ├── wyscout_shots_processed.csv
│       └── player_injuries_processed.csv
│
├── notebooks/
│   ├── cycle1/                  # Match outcome prediction
│   │   ├── cycle1_exploration.ipynb
│   │   ├── cycle1_preprocessing.ipynb
│   │   ├── cycle1_modelling.ipynb
│   │   ├── cycle1_tuning.ipynb
│   │   ├── cycle1_explainability.ipynb     # SHAP analysis — saves plots to docs/
│   │   └── chronological/
│   │       ├── cycle1_modelling_chronological.ipynb
│   │       └── cycle1_tuning_chronological.ipynb  ← DEPLOYED MODEL SOURCE
│   ├── cycle2/                  # Expected goals
│   │   ├── cycle2_exploration_wyscout.ipynb
│   │   ├── cycle2_preprocessing_wyscout.ipynb
│   │   ├── cycle2_modelling.ipynb
│   │   ├── cycle2_tuning.ipynb             ← DEPLOYED MODEL SOURCE
│   │   └── cycle2_explainability.ipynb
│   └── cycle3/                  # Injury risk
│       ├── cycle3_exploration_injuries.ipynb
│       ├── cycle3_preprocessing_injuries.ipynb
│       ├── cycle3_modelling.ipynb
│       ├── cycle3_tuning.ipynb
│       ├── cycle3_explainability.ipynb
│       └── chronological/
│           ├── cycle3_modelling_chronological.ipynb
│           └── cycle3_tuning_chronological.ipynb   ← DEPLOYED MODEL SOURCE
│
├── models/
│   ├── cycle1/
│   │   ├── cycle1_xgb_best.pkl            # XGBoost tuned (chronological)
│   │   └── cycle1_feature_cols.pkl        # 33 feature names in order
│   ├── cycle2/
│   │   ├── cycle2_best_model.pkl          # XGBoost tuned (random)
│   │   ├── cycle2_scaler.pkl              # StandardScaler fit on training data
│   │   └── cycle2_feature_cols.pkl        # 9 feature names in order
│   └── cycle3/
│       ├── cycle3_best_model.pkl          # LightGBM tuned (chronological)
│       ├── cycle3_scaler.pkl              # StandardScaler fit on training data
│       └── cycle3_feature_cols.pkl        # 17 feature names in order
│
├── api/                         # FastAPI backend
│   ├── main.py                  # App factory, lifespan startup
│   ├── dependencies.py          # FastAPI dependency injection
│   ├── ml/
│   │   ├── pipeline.py          # Inference-time feature engineering
│   │   └── feature_store.py     # Per-team rolling stats cache
│   ├── routers/
│   │   ├── predict.py           # POST /predict
│   │   ├── explain.py           # POST /explain
│   │   ├── health.py            # GET /health/ready
│   │   ├── models.py            # GET /models
│   │   └── teams.py             # GET /teams
│   ├── schemas/
│   │   ├── prediction.py        # Request/response Pydantic models
│   │   ├── explanation.py       # SHAP explanation schemas
│   │   ├── team.py              # Team info schemas
│   │   └── model_info.py        # Model metadata schemas
│   └── services/
│       ├── model_registry.py    # Loads & caches all .pkl artefacts at startup
│       ├── prediction_service.py # Business logic for all prediction types
│       ├── explanation_service.py # SHAP computation service
│       └── team_service.py      # Team lookup service
│
├── dashboard/                   # Streamlit frontend
│   ├── app.py                   # Landing page
│   ├── components/
│   │   ├── api_client.py        # HTTP client wrapping the FastAPI backend
│   │   └── charts.py            # Plotly chart builders
│   └── pages/
│       ├── 1_Match_Predictor.py
│       ├── 2_xG_Calculator.py
│       └── 3_Injury_Risk.py
│
└── docs/                        # Saved plots (SHAP, ROC curves, model comparisons)
    ├── cycle1_shap_global_importance.png
    ├── cycle1_shap_summary_side_by_side.png
    ├── cycle1_shap_per_class.png
    ├── cycle1_shap_waterfall.png
    ├── cycle1_model_comparison.png
    ├── cycle2_roc_curve.png
    ├── cycle2_roc_curves_detailed.png
    ├── cycle2_roc_baseline.png
    └── cycle3_roc_baseline.png
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
pip install -r requirements.txt
```

### 2. Run the notebooks (in order per cycle)

Each cycle follows the same sequence:
```
exploration → preprocessing → modelling → tuning → explainability
```

The `chronological/` subfolder notebooks for Cycles 1 and 3 produce the deployed models.

### 3. Start the API

```bash
.venv/bin/python -m uvicorn api.main:app --reload --port 8000
```

Interactive API docs available at: `http://localhost:8000/docs`

### 4. Start the dashboard

```bash
.venv/bin/streamlit run dashboard/app.py
```

Dashboard available at: `http://localhost:8501`

---

## API Overview

All predictions go through a single endpoint:

```
POST /predict
```

Set the `model` field to one of:

| Value | Description |
|-------|-------------|
| `"match"` | Predict Home Win / Draw / Away Win |
| `"xg"` | Predict xG (goal probability) for a shot |
| `"injury"` | Predict player injury risk |
| `"pipeline"` | Run all three models and combine outputs |

```
POST /explain
```

Returns SHAP feature importance values for any prediction.

---

## Model Performance Summary

| Model | Task | Algorithm | Metric | Value | Baseline |
|-------|------|-----------|--------|-------|----------|
| Cycle 1 | Match outcome | XGBoost (tuned) | Accuracy | 52.85% | 46.35% (dummy) |
| Cycle 2 | xG (shot → goal) | XGBoost (tuned) | AUC-ROC | 0.8183 | 0.50 (dummy) |
| Cycle 3 | Injury risk | LightGBM (tuned) | AUC-ROC | 0.6800 | 0.50 (dummy) |

### Important note on evaluation splits

- **Cycle 1** uses a **chronological split** (train on earlier seasons, test on the most recent 20% of matches). This mirrors real deployment: predicting future matches from past data. The random-split XGBoost achieves a near-identical 52.78%, confirming no significant temporal leakage.
- **Cycle 2** uses a **random stratified split** (no temporal ordering in shot data; stratification preserves the 10.8% goal rate).
- **Cycle 3** uses a **chronological split** by `start_year`. Random-split results are also available but are considered less realistic.

---

## Key Design Decisions

### Why three separate models instead of one?
Each prediction task uses a fundamentally different dataset, target variable, and feature set. Match outcome needs team-level aggregate statistics; xG needs individual shot geometry; injury risk needs player physical history. A single model cannot serve all three purposes.

### Why XGBoost for Cycles 1 and 2?
XGBoost consistently outperformed Logistic Regression, Random Forest, and LightGBM on both tasks after tuning. It handles heterogeneous tabular features well and provides native SHAP support for explainability.

### Why LightGBM for Cycle 3?
LightGBM achieved the highest AUC (0.6800) on the chronological test set, beating XGBoost (0.6723), Random Forest (0.6668), and Logistic Regression (0.6263). It handles the smaller dataset (1,301 rows) more efficiently than XGBoost with its leaf-wise tree growth.

### Why SHAP?
SHAP (SHapley Additive exPlanations) provides mathematically rigorous, per-prediction feature attributions. This transforms black-box model outputs into transparent explanations that football analysts and medical staff can act on.

### Why a pipeline mode?
Real match prediction benefits from knowing about shot quality and squad fitness. The pipeline mode chains all three models to produce a richer, adjusted probability estimate that no single model could provide alone.

---

## Dependencies

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy |
| ML | scikit-learn, xgboost, lightgbm |
| Explainability | shap |
| Visualisation | matplotlib, seaborn, plotly |
| API | fastapi, uvicorn, pydantic |
| Dashboard | streamlit, requests |
| Persistence | joblib |
| Notebooks | jupyter, ipykernel |

---

## What Has Changed Since the Initial README

The original README was a placeholder. This updated version reflects:

1. **Three fully trained and deployed models** — all `.pkl` artefacts are present in `models/`
2. **Explainability notebooks** for all three cycles (Cycle 1 was original; Cycles 2 and 3 added)
3. **Pipeline mode** in the API — chains all three models with probability adjustment
4. **Streamlit dashboard** with three pages, each connecting to the API
5. **SHAP endpoints** — `/explain` computes and returns per-prediction feature contributions
6. **Feature store** — caches team rolling stats at API startup for fast match predictions
7. **Chronological evaluation** — Cycles 1 and 3 now report honest deployment-realistic AUC/accuracy

---

*Documentation generated from project state as of May 2026.*
