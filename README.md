# Football Predictor

A full-stack football analytics platform built around three independent machine learning cycles. A FastAPI backend serves predictions and SHAP explanations; a Streamlit dashboard makes them interactive. All models use **chronological train/test splits** to prevent data leakage — the test set always comes from later dates than the training set.

---

## What it predicts

| Cycle | Task | Algorithm | Key metric |
|-------|------|-----------|------------|
| 1 | **Match outcome** — Home Win / Draw / Away Win | XGBoost (tuned) | 52.85% accuracy (chronological hold-out) |
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
│   ├── cycle1/                      # Premier League: preprocessing, EDA, modelling, tuning, explainability
│   │   └── chronological/           # Time-split variants of modelling + tuning (deployed source)
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
| `premier_league_matches.csv` | Premier League 2000-2018 match results + season-to-date features (form, points, streaks, last-5 results, goal difference) — 6,840 matches across 18 seasons, 44 teams | Cycle 1 (training + feature store) |
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
1. `notebooks/cycle1/cycle1_preprocessing.ipynb`
2. `notebooks/cycle1/chronological/cycle1_tuning_chronological.ipynb` → saves to `models/cycle1/` (deployed source)

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
  "home_team_id": 24,
  "away_team_id": 23,
  "mw": 20
}
```

`home_team_id` / `away_team_id` are alphabetically encoded over all PL teams 2000-2018 (range 0-43; e.g. Arsenal=0, Liverpool=23, Man City=24). `mw` is the matchweek (1-38) and defaults to the latest matchweek in the dataset. Optional: pass `home_stats` and `away_stats` dicts with the team's season-to-date snapshot to override the feature store.

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
| `GET` | `/teams` | List all 44 Premier League teams (2000-2018) with IDs |
| `GET` | `/teams/{id}` | Single team with latest snapshot stats |
| `GET` | `/models` | Metadata for all three deployed models |
| `GET` | `/models/{name}/compare` | All evaluated variants for a given cycle (`match`, `xg`, `injury`) |

---

## How each cycle works

### Cycle 1 — Match outcome

**Feature set (33):** Pre-engineered season-to-date stats from the Premier League dataset:

- Identity: `HomeTeam`, `AwayTeam` (alphabetically encoded over 44 teams)
- Goals: `HTGS`, `ATGS`, `HTGC`, `ATGC` (scored / conceded so far this season)
- Points and goal difference: `HTP`, `ATP`, `HTGD`, `ATGD`, `DiffPts`
- Form: `HM1..HM5`, `AM1..AM5` (last 5 results, encoded 0=L, 1=D, 3=W), `HTFormPts`, `ATFormPts`, `DiffFormPts`
- Streaks: `HTWinStreak3/5`, `HTLossStreak3/5`, `ATWinStreak3/5`, `ATLossStreak3/5`
- `MW`: matchweek (1-38)

**Train/test split:** Chronological by season — training on the earliest 80% of matches (2000-2014), testing on the most recent 20% (2015-2018). This reflects real deployment: the model never sees future information during training.

**Feature vector at inference:** The feature store (built from the processed PL CSV at API startup) caches the latest snapshot of each team's season-to-date stats. Users can override these via the Manual or CSV upload modes on the dashboard for hypothetical or future fixtures.

**Result:** 52.85% test accuracy vs ~46% dummy baseline (+6.5pp). 5,472 training matches; XGBoost dominates the Home Win class (precision 0.54, recall 0.86) but draws remain hard (recall 0.04) — a known limitation across published football-prediction work.

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

A random split shuffles matches from all seasons into train and test. This means the model can learn patterns from 2017 matches while being tested on 2010 matches — it has "seen the future". For time-series data like football seasons, this inflates performance estimates.

A chronological split trains on early seasons and tests on later ones, matching how a deployed model actually operates. For Cycle 1, the random and chronological tuned XGBoost numbers are very close (52.78% vs 52.85%), but the chronological number is the one we trust — random-split estimates can hide season-specific drift.

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
- **Cycle 1 match:** Extend the PL dataset past 2018 (current data ends with the 2017-18 season). Adding modern seasons would also let the feature store reflect current squads.
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
