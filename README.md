# Football Predictor

A full-stack football analytics platform built around three independent machine learning cycles. A FastAPI backend serves predictions and SHAP explanations; a Streamlit dashboard makes them interactive. All models use **chronological train/test splits** to prevent data leakage — the test set always comes from later dates than the training set.

---

## What it predicts

| Cycle | Task | Algorithm | Key metric |
|-------|------|-----------|------------|
| 1 | **Match outcome** — Home Win / Draw / Away Win | XGBoost (tuned) | 50.22% accuracy (chronological hold-out) |
| 2 | **Expected Goals (xG)** — probability a shot scores | XGBoost (tuned) | AUC 0.8342 (chronological matchId split) |
| 3 | **Player injury risk** — high risk = 28+ days missed | XGBoost (tuned) | AUC 0.6723 (chronological year split) |

Why AUC instead of accuracy for Cycles 2 & 3: 89% of shots are not goals and 70% of players are "high injury", so a dummy model achieves high accuracy by predicting the majority class. AUC-ROC measures whether the model can actually rank the positive class above the negative class.

---

## Project structure

```
FootballPredictor/
├── config.py                        # Shared path constants for all notebooks
│
├── api/                             # FastAPI backend
│   ├── main.py                      # App factory + lifespan startup
│   ├── routers/
│   │   ├── predict.py               # POST /predict
│   │   ├── explain.py               # POST /explain
│   │   ├── models.py                # GET  /models, /models/{name}/compare
│   │   ├── teams.py                 # GET  /teams, /teams/{id}
│   │   └── health.py                # GET  /health/ready
│   ├── services/
│   │   ├── model_registry.py        # Loads all .pkl artefacts at startup
│   │   ├── prediction_service.py    # Business logic for all three cycles
│   │   ├── explanation_service.py   # SHAP TreeExplainer / LinearExplainer
│   │   └── team_service.py          # Team lookup + rolling stats
│   ├── ml/
│   │   ├── feature_store.py         # Rolling 5-match stats per team (built at startup)
│   │   └── pipeline.py              # Feature vector builders (match / xG / injury)
│   └── schemas/                     # Pydantic request/response models
│
├── dashboard/                       # Streamlit frontend
│   ├── app.py                       # Landing page
│   ├── pages/
│   │   ├── 1_Match_Predictor.py     # Auto / Manual / CSV stats input
│   │   ├── 2_xG_Calculator.py       # Interactive pitch map
│   │   ├── 3_Injury_Risk.py         # Player profile form
│   │   └── 4_Model_Explorer.py      # All model comparison tables
│   └── components/
│       ├── api_client.py            # HTTP wrappers (Streamlit → FastAPI)
│       └── charts.py                # Plotly chart builders
│
├── models/
│   ├── cycle1/                      # cycle1_xgb_best.pkl, cycle1_scaler.pkl, cycle1_feature_cols.pkl
│   ├── cycle2/                      # cycle2_best_model.pkl, cycle2_scaler.pkl, cycle2_feature_cols.pkl
│   └── cycle3/                      # cycle3_best_model.pkl, cycle3_scaler.pkl, cycle3_feature_cols.pkl
│
├── notebooks/
│   ├── cycle1/
│   │   ├── premier_league/          # PL preprocessing, EDA, modelling, tuning, explainability
│   │   │   └── chronological/       # Time-split variants of modelling + tuning
│   │   └── skysports/               # SkySports preprocessing, EDA, feature engineering
│   ├── cycle2/
│   │   ├── cycle2_preprocessing_wyscout.ipynb
│   │   ├── cycle2_exploration_wyscout.ipynb
│   │   ├── cycle2_modelling.ipynb
│   │   ├── cycle2_tuning.ipynb
│   │   └── chronological/
│   └── cycle3/
│       ├── cycle3_preprocessing_injuries.ipynb
│       ├── cycle3_exploration_injuries.ipynb
│       ├── cycle3_modelling.ipynb
│       ├── cycle3_tuning.ipynb
│       └── chronological/
│
└── data/
    ├── raw/                         # Source datasets (not modified)
    └── processed/                   # Cleaned CSVs produced by preprocessing notebooks
```

---

## Datasets

| Dataset | Source | Used by |
|---------|--------|---------|
| `premier_league_matches.csv` | Premier League 2016-23 match results + basic stats | Cycle 1 Dataset 1 |
| `skysports_match_stats.csv` | SkySports 2022-23 detailed per-match stats (25 PL teams) | Cycle 1 Dataset 2 + feature store |
| `events_England.json` + `playerank.json` + `players.json` | Wyscout England event data | Cycle 2 |
| `player_injuries.csv` | Multi-season player injury records + FIFA attributes | Cycle 3 |

---

## Getting started

### Requirements

- Python 3.10+
- macOS: `brew install libomp` (required by XGBoost)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the models (first time)

Run the tuning notebooks in order to produce the `.pkl` files. Every notebook finds the project root automatically, so you can run them from any working directory.

**Cycle 1 — Match outcome:**
1. `notebooks/cycle1/skysports/cycle1_preprocessing_skysports_match_stats.ipynb`
2. `notebooks/cycle1/skysports/cycle1_feature_engineering_skysports.ipynb`
3. `notebooks/cycle1/premier_league/cycle1_preprocessing_premier_league_matches.ipynb`
4. `notebooks/cycle1/premier_league/cycle1_tuning.ipynb` → saves to `models/cycle1/`

**Cycle 2 — Expected Goals:**
1. `notebooks/cycle2/cycle2_preprocessing_wyscout.ipynb`
2. `notebooks/cycle2/cycle2_tuning.ipynb` → saves to `models/cycle2/`

**Cycle 3 — Injury Risk:**
1. `notebooks/cycle3/cycle3_preprocessing_injuries.ipynb`
2. `notebooks/cycle3/cycle3_tuning.ipynb` → saves to `models/cycle3/`

> The `chronological/` notebooks in each cycle are the deployed versions — use those for the final saved models.

### Run the API

```bash
.venv/bin/python -m uvicorn api.main:app --port 8000 --reload
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Run the dashboard

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501). The sidebar shows whether the API is reachable.

---

## API reference

All requests go to `http://localhost:8000`.

### `POST /predict`

Unified endpoint for all three cycles. The `model` field selects the prediction type.

**Cycle 1 — match outcome:**
```json
{
  "model": "match",
  "home_team_id": 5,
  "away_team_id": 2,
  "attendance": 52000
}
```

Optional: pass `home_stats` and `away_stats` dicts with rolling averages to override the feature store (useful for future fixtures where stored data is stale).

**Cycle 2 — expected goals:**
```json
{
  "model": "xg",
  "X": 88.0,
  "Y": 50.0,
  "left_foot": 0,
  "right_foot": 1,
  "header": 0,
  "first_half": 1,
  "player_rank": 7.2
}
```

Distance and angle to goal are computed automatically from X/Y.

**Cycle 3 — injury risk:**
```json
{
  "model": "injury",
  "age": 27,
  "height_cm": 181,
  "weight_kg": 76,
  "bmi": 23.2,
  "pace": 72,
  "physic": 75,
  "fifa_rating": 78,
  "work_rate_numeric": 2,
  "position_numeric": 3,
  "cumulative_minutes_played": 8200,
  "cumulative_games_played": 95,
  "cumulative_days_injured": 45,
  "minutes_per_game_prev_seasons": 72.0,
  "avg_days_injured_prev_seasons": 8.5,
  "avg_games_per_season_prev_seasons": 28.0,
  "significant_injury_prev_season": 0,
  "season_days_injured_prev_season": 12.0
}
```

**Response (all cycles):**
```json
{
  "prediction": "Home Win",
  "probabilities": { "home_win": 0.52, "draw": 0.28, "away_win": 0.20 },
  "confidence": "medium",
  "model_used": "match"
}
```

### `POST /explain`

Same request body as `/predict` plus an optional `top_n` field (default 10). Returns SHAP values ranked by absolute impact.

```json
{
  "base_value": -0.1234,
  "top_features": [
    { "feature": "avg_shots_on_target_5_home", "value": 4.2, "shap_value": 0.31, "impact": "increases_risk" },
    ...
  ],
  "all_features": [ ... ]
}
```

### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health/ready` | Returns 200 once models and feature store are loaded |
| `GET` | `/teams` | List all 25 Premier League teams with IDs |
| `GET` | `/teams/{id}` | Single team with latest rolling stats |
| `GET` | `/models` | Metadata for all three deployed models |
| `GET` | `/models/{name}/compare` | All evaluated variants for a given cycle (`match`, `xg`, `injury`) |

---

## How each cycle works

### Cycle 1 — Match outcome

**Feature engineering:** SkySports per-match stats (possession, shots, shots on target, pass accuracy, tackles, corners, fouls, yellow cards) are aggregated into 5-match rolling averages per team using `shift(1).rolling(5, min_periods=1).mean()`. The shift prevents the current match from leaking into its own features. Home and away rolling stats are joined with Premier League result labels (H/D/A).

**Train/test split:** Chronological by date — training on earlier seasons, testing on the most recent season. This reflects real deployment: the model never sees future information during training.

**Feature vector at inference:** The feature store (built from the SkySports CSV at API startup) supplies the latest rolling averages for any team. Users can override these via the Manual or CSV upload modes on the dashboard.

**Result:** 50.22% test accuracy vs 48.89% dummy baseline. The legacy 57.33% number used a random split with leaky cross-validation scaling — it inflated the estimate by about 7 percentage points.

### Cycle 2 — Expected Goals

**Data:** Wyscout England event data (~8,451 shots). Only pre-shot features are used — X/Y coordinates, foot type, match half, player rank. Post-shot tags (where the ball ended up in the net) are excluded because they are not knowable before the shot.

**Feature engineering:** Distance (metres) and angle (degrees) to goal are computed from X/Y. Distance = Euclidean from shot position to goal centre. Angle = arctan between the two posts from the shot position.

**Train/test split:** Chronological by matchId (lower IDs = earlier matches).

**Result:** AUC 0.8342 vs 0.5 dummy baseline. The 9:1 class imbalance (90% non-goals) means accuracy alone is misleading.

### Cycle 3 — Player injury risk

**Target:** Binary — 1 if a player missed 28+ days in a season (significant injury). The 28-day threshold follows the sports-medicine definition of a long-term injury. Using any injury at all would produce a near-trivially imbalanced target (≈99.9% injured).

**Features (17):** Physical attributes (height, weight, BMI, FIFA pace/physic/rating), demographics (age, position, work rate), and injury history (career totals, previous-season averages, whether they had a significant injury last season).

**Train/test split:** Chronological by `start_year` — earlier seasons train, later seasons test.

**Result:** AUC 0.6723, at the top of the published sports-science range (0.60–0.70) for injury prediction models. The model cannot account for training load, contact events, or pitch conditions — the primary causal injury factors — so it should be treated as a screening tool rather than a definitive risk score.

---

## Methodology notes

### Why chronological splitting matters

A random split shuffles matches from all seasons into train and test. This means the model can learn patterns from 2022 matches while being tested on 2020 matches — it has "seen the future". For time-series data like football seasons, this inflates performance estimates.

A chronological split trains on early seasons and tests on later ones, matching how a deployed model actually operates. The difference is significant: for Cycle 1, random-split XGBoost tuned to 57.33% accuracy; the same model on a chronological split achieves 50.22%.

### Why sklearn Pipeline for cross-validation

Early versions fitted the scaler on the full training set before cross-validation, then applied it inside each fold. This is data leakage — the scaler has seen validation-fold statistics. Wrapping `StandardScaler + XGBClassifier` in a `Pipeline` ensures scaling is re-fitted on each fold's training portion only.

### SHAP explanations

- **Cycles 1 & 2 (XGBoost):** `shap.TreeExplainer` — exact Shapley values via the tree structure, no approximation needed.
- **Cycle 3 (XGBoost):** Same TreeExplainer. The API handles both the legacy 2D output format and the modern 3D `(samples, features, classes)` format from newer SHAP versions.
- The dashboard shows SHAP as a ranked bar chart. The base value is the model's prior (expected prediction before seeing any features).

---

## Adding more data

The highest-priority improvements identified during development:

- **Cycle 2 xG:** Add France, Germany, Italy, Spain Wyscout data (~50,000 shots total vs current 8,451). Expected AUC improvement to ~0.85+.
- **Cycle 1 match:** Extend SkySports scraping beyond the 2022-23 season. The rolling-window features are dataset-agnostic — more seasons = better signal.
- **Cycle 3 injury:** Incorporate training load data (GPS minutes, sprint counts). Physical workload is the largest missing causal factor.

---

## Shared path config

All notebooks import paths from `config.py` at the project root. Adding a new dataset or changing a directory name only requires editing one file:

```python
from config import Paths

df = pd.read_csv(str(Paths.PL_MATCHES_PROCESSED))
joblib.dump(model, str(Paths.C1_MODEL))
```

The import snippet in every notebook finds the project root by walking up from `os.getcwd()` until it finds the `data/` directory, so notebooks run correctly from any working directory regardless of their depth in the folder hierarchy.
