# Football Predictor

Three machine-learning models for football analytics, served through a single FastAPI backend and a Streamlit dashboard.

| Cycle | Task | Notebook folder | Deployed model | Metric |
|---|---|---|---|---|
| 1 | **Match outcome** — Home Win / Draw / Away Win for a Premier League fixture | [`notebooks/cycle1/`](notebooks/cycle1/README.md) | XGBoost (tuned, chronological) | 52.85% accuracy |
| 2 | **Expected Goals (xG)** — probability that a single shot becomes a goal | [`notebooks/cycle2/`](notebooks/cycle2/README.md) | XGBoost (tuned, random) | AUC 0.8183 |
| 3 | **Injury risk** — probability that a player misses 28+ days this season | [`notebooks/cycle3/`](notebooks/cycle3/README.md) | LightGBM (tuned, chronological) | AUC 0.68 |

Each cycle is self-contained: its own dataset, preprocessing, exploration, modelling, tuning, and saved artefacts. The API loads all three at startup and exposes them through one unified surface.

---

## Repository layout

```
FootballPredictor/
├── api/                    # FastAPI backend
│   ├── main.py             # app factory + lifespan (loads models at startup)
│   ├── routers/            # /predict, /explain, /models, /teams, /health
│   ├── schemas/            # Pydantic I/O contracts (discriminated unions)
│   ├── services/           # model_registry, prediction, explanation, teams
│   └── ml/                 # feature_store + per-cycle feature builders
├── dashboard/              # Streamlit UI
│   ├── app.py              # landing page
│   ├── pages/              # Match Predictor, xG Calculator, Injury Risk
│   └── components/         # api_client, chart helpers
├── notebooks/
│   ├── cycle1/             # match outcome — see notebooks/cycle1/README.md
│   ├── cycle2/             # xG            — see notebooks/cycle2/README.md
│   └── cycle3/             # injury risk   — see notebooks/cycle3/README.md
├── data/
│   ├── raw/                # source CSV/JSON files
│   └── processed/          # ML-ready CSVs written by preprocessing notebooks
├── models/                 # saved .pkl artefacts (one folder per cycle)
├── docs/                   # SHAP plots, ROC curves, etc.
├── config.py               # Paths.* — single source of truth for file locations
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (Python 3.11+ recommended)
python -m venv .venv
source .venv/bin/activate              # macOS / Linux
# .venv\Scripts\activate                # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the system

The API and dashboard are two separate processes. Start the API first.

```bash
# Terminal 1 — API on http://localhost:8000
python -m uvicorn api.main:app --port 8000 --reload

# Terminal 2 — Dashboard on http://localhost:8501
streamlit run dashboard/app.py
```

Once both are up:

- Dashboard: <http://localhost:8501>
- API docs (Swagger): <http://localhost:8000/docs>
- API docs (ReDoc): <http://localhost:8000/redoc>
- Health check: <http://localhost:8000/health/ready>

---

## API surface

A single `POST /predict` endpoint dispatches to the right cycle using the `model` discriminator (`"match"`, `"xg"`, or `"injury"`). `POST /explain` mirrors the same contract and returns SHAP feature impacts.

```bash
# Match outcome
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"match","home_team_id":24,"away_team_id":23,"mw":20}'

# xG
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"xg","X":85,"Y":50,"left_foot":0,"right_foot":1,"header":0,"first_half":1,"player_rank":6.5}'

# Injury risk (17 features — see api/schemas/prediction.py)
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"injury","height_cm":181,"weight_kg":76,"pace":72,"physic":75,"fifa_rating":78,"age":26,"bmi":23.2,"work_rate_numeric":2,"position_numeric":3,"cumulative_minutes_played":8200,"cumulative_games_played":95,"cumulative_days_injured":45,"minutes_per_game_prev_seasons":72,"avg_days_injured_prev_seasons":8.5,"avg_games_per_season_prev_seasons":28,"significant_injury_prev_season":0,"season_days_injured_prev_season":12}'
```

---

## Reproducing the models

Each cycle's `README.md` lists the notebook order. The general flow:

1. **Exploration** — inspect the raw file, identify schema issues and leakage.
2. **Preprocessing** — write a clean CSV to `data/processed/`.
3. **Modelling** — train baselines on a random split.
4. **Tuning** — `RandomizedSearchCV` over candidate algorithms.
5. **Chronological tuning** (cycles 1 and 3) — same search, time-respecting split. **Saves the deployed model.**
6. **Explainability** (cycle 1) — SHAP analysis on the deployed model.

`config.py` exposes a `Paths` class with absolute paths to every data and model file. Notebooks discover the project root automatically; you don't need to edit paths when moving the repo.

---

## Methodology notes

- **Splits.** Cycles 1 and 3 deploy the chronological-split winner. The random split is kept for comparison only — it leaks future seasons into training and overstates deployment-time accuracy.
- **Metrics.** Cycle 1 uses accuracy (3-class, roughly balanced after Home-Win advantage). Cycles 2 and 3 use AUC-ROC because both have heavy class imbalance (89:11 and 70:30) where accuracy is misleading.
- **Scaling.** Tree models (XGBoost / RF / LightGBM) don't need scaled inputs. Cycle 1's deployed XGBoost is therefore served on raw features and ships without a scaler. Cycles 2 and 3 still ship a `StandardScaler` because their pipelines compared against a Logistic Regression baseline that does need scaling.
- **Explainability.** SHAP `TreeExplainer` is used for all tree models (XGBoost / LightGBM / RF). `LinearExplainer` is reserved for linear baselines.

---

## Tech stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy |
| Models | scikit-learn, xgboost, lightgbm |
| Explainability | shap |
| Visualisation | matplotlib, seaborn, plotly |
| Backend | fastapi, uvicorn, pydantic |
| Frontend | streamlit |
| Persistence | joblib |
| Notebooks | jupyter, ipykernel, nbconvert |

Full pinned list: [`requirements.txt`](requirements.txt).
