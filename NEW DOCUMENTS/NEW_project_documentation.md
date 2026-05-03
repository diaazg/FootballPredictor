# Full Project Documentation — Smart Football Analytics

**Project title:** Smart Football Analytics: Predicting Matches, Goals, and Injuries with Machine Learning  
**Student:** Abdulaziz Alaskar (200612007)  
**Institution:** Queen Mary University of London (QMUL)  
**Degree:** BSc Computer Science & Artificial Intelligence  
**Supervisor:** Tayyab Ahmad Ansari  
**Academic year:** 2025–2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Aims and Objectives](#2-aims-and-objectives)
3. [Architecture Overview](#3-architecture-overview)
4. [Folder Structure Explained](#4-folder-structure-explained)
5. [Datasets](#5-datasets)
6. [Machine Learning Pipeline (Per Cycle)](#6-machine-learning-pipeline-per-cycle)
7. [Model Artefacts and Persistence](#7-model-artefacts-and-persistence)
8. [FastAPI Backend](#8-fastapi-backend)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [The Unified Pipeline Mode](#10-the-unified-pipeline-mode)
11. [SHAP Explainability System](#11-shap-explainability-system)
12. [Configuration System](#12-configuration-system)
13. [Design Decisions and Rationale](#13-design-decisions-and-rationale)
14. [Limitations](#14-limitations)
15. [Future Improvements](#15-future-improvements)

---

## 1. Project Overview

This project builds a **unified football analytics prediction system** that combines three machine learning models into a single deployable application. The three models address distinct analytical questions:

| Cycle | Question | Type | Best Metric |
|-------|----------|------|-------------|
| 1 | Will this team win, draw, or lose? | 3-class classification | 52.85% accuracy |
| 2 | What is the goal probability of this shot? | Binary classification | AUC-ROC 0.8183 |
| 3 | Is this player at high injury risk this season? | Binary classification | AUC-ROC 0.6800 |

All three models are served through a single REST API (FastAPI) and visualised in an interactive web dashboard (Streamlit). A "pipeline" mode chains all three models: shot data feeds the xG model, player data feeds the injury model, and their combined outputs adjust the match model's probability estimates.

The system follows a consistent pattern across all cycles:
```
Raw Data → Exploration → Preprocessing → Modelling → Tuning → Explainability → API → Dashboard
```

---

## 2. Aims and Objectives

### Primary Aim
Build a deployable, explainable football analytics system that demonstrates the application of supervised machine learning to three connected football prediction tasks.

### Objectives
1. **Cycle 1:** Apply classification methods to predict Premier League match outcomes; achieve accuracy meaningfully above the naive majority-class baseline.
2. **Cycle 2:** Build an xG model using shot geometry and player quality; demonstrate that AUC-ROC is the correct metric for imbalanced binary classification.
3. **Cycle 3:** Model player injury risk from pre-season data; avoid data leakage from in-season statistics.
4. **Integration:** Expose all three models through a unified API with a single `model` discriminator field and a pipeline mode.
5. **Explainability:** Provide SHAP-based feature attributions for every prediction, making the system transparent to non-technical users.
6. **Evaluation integrity:** Use chronological train/test splits for Cycles 1 and 3 to ensure evaluation metrics reflect realistic deployment performance.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT DASHBOARD                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  1_Match         │  │  2_xG_Calculator │  │  3_Injury_Risk  │  │
│  │  Predictor.py    │  │  .py             │  │  .py            │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘  │
│           │ HTTP requests        │                     │            │
└───────────┼──────────────────────┼─────────────────────┼────────────┘
            │                      │                     │
            ▼                      ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  POST /predict    POST /explain    GET /health    GET /teams │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                  prediction_service.py                        │   │
│  │  predict_match()  predict_xg()  predict_injury()  pipeline() │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                   model_registry.py                           │   │
│  │     match: XGBoost   xg: XGBoost   injury: LightGBM          │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │  ml/pipeline.py                ml/feature_store.py            │   │
│  │  build_match_vector()          get_team_stats(team_id)        │   │
│  │  build_xg_vector()             Rolling stats cache (in-memory)│   │
│  │  build_injury_vector()                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             │ (loaded at startup via lifespan)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL ARTEFACTS (models/)                     │
│  cycle1/cycle1_xgb_best.pkl          cycle1/cycle1_feature_cols.pkl  │
│  cycle2/cycle2_best_model.pkl        cycle2/cycle2_scaler.pkl         │
│  cycle2/cycle2_feature_cols.pkl                                       │
│  cycle3/cycle3_best_model.pkl        cycle3/cycle3_scaler.pkl         │
│  cycle3/cycle3_feature_cols.pkl                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Folder Structure Explained

### `config.py`
The project root contains a single shared configuration file, `config.py`, which defines the `Paths` class. Every notebook and API module imports from this to resolve file paths. This prevents hard-coded paths and makes the project portable.

```python
from config import Paths, ensure_dirs
df = pd.read_csv(str(Paths.WYSCOUT_PROCESSED))
```

The `ensure_dirs()` function creates all `models/` subdirectories if they do not exist.

### `data/`

| Subfolder | Contents |
|-----------|----------|
| `raw/` | Original source datasets. Never modified by the project. |
| `processed/` | Feature-engineered CSVs written by preprocessing notebooks. |

The raw data is treated as immutable. All transformations produce new files in `processed/`.

### `notebooks/`

Organised by cycle, each following the same progression:

```
exploration.ipynb → preprocessing.ipynb → modelling.ipynb → tuning.ipynb → explainability.ipynb
```

Cycles 1 and 3 also have a `chronological/` subfolder containing notebooks that use time-ordered train/test splits. These produce the deployed models.

**Why split random vs. chronological?** The random-split notebooks establish baseline comparisons using the conventional approach familiar from academic ML courses. The chronological notebooks answer the more important question: *how does the model perform on future data it has never seen?* This mirrors actual deployment conditions.

### `models/`

Stores serialised model artefacts as `.pkl` files (via `joblib`). Each cycle has its own subdirectory:

| File | Contents | Cycle |
|------|----------|-------|
| `cycle1/cycle1_xgb_best.pkl` | Trained XGBoost (33 features) | 1 |
| `cycle1/cycle1_feature_cols.pkl` | Ordered list of 33 feature names | 1 |
| `cycle2/cycle2_best_model.pkl` | Trained XGBoost (9 features, scaled) | 2 |
| `cycle2/cycle2_scaler.pkl` | `StandardScaler` fit on C2 training data | 2 |
| `cycle2/cycle2_feature_cols.pkl` | Ordered list of 9 feature names | 2 |
| `cycle3/cycle3_best_model.pkl` | Trained LightGBM (17 features, scaled) | 3 |
| `cycle3/cycle3_scaler.pkl` | `StandardScaler` fit on C3 training data | 3 |
| `cycle3/cycle3_feature_cols.pkl` | Ordered list of 17 feature names | 3 |

Three artefacts per cycle are necessary because:
- The **model** contains the learned weights
- The **scaler** contains the mean/std computed from training data (must be applied identically at inference)
- The **feature list** ensures features are presented to the model in exactly the same column order as during training

### `api/`

The backend is organised in layers:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Routing | `routers/` | URL pattern definitions, HTTP method bindings |
| Business logic | `services/` | Prediction computation, SHAP computation, data assembly |
| ML pipeline | `ml/` | Feature engineering at inference time |
| Data contracts | `schemas/` | Pydantic models for request/response validation |

### `dashboard/`

| File | Purpose |
|------|---------|
| `app.py` | Landing page; API status check |
| `components/api_client.py` | HTTP wrapper around all API endpoints |
| `components/charts.py` | Reusable Plotly chart builders |
| `pages/1_Match_Predictor.py` | Match outcome prediction UI |
| `pages/2_xG_Calculator.py` | xG shot calculator UI |
| `pages/3_Injury_Risk.py` | Player injury risk UI |

Streamlit's multi-page app convention automatically creates a navigation sidebar from the `pages/` directory, with pages ordered by their numeric prefix.

### `docs/`

Saved visualisation outputs from the explainability notebooks:

| File | Source notebook | Content |
|------|----------------|---------|
| `cycle1_shap_global_importance.png` | `cycle1_explainability.ipynb` | Mean \|SHAP\| bar chart |
| `cycle1_shap_summary_side_by_side.png` | `cycle1_explainability.ipynb` | 3-class beeswarm |
| `cycle1_shap_per_class.png` | `cycle1_explainability.ipynb` | Per-class importance bars |
| `cycle1_shap_waterfall.png` | `cycle1_explainability.ipynb` | Single-match waterfall |
| `cycle1_model_comparison.png` | `cycle1_modelling*.ipynb` | Accuracy comparison chart |
| `cycle2_roc_curve.png` | `cycle2_tuning.ipynb` | ROC curve (tuned XGBoost) |
| `cycle2_roc_curves_detailed.png` | `cycle2_tuning.ipynb` | All models ROC comparison |
| `cycle2_roc_baseline.png` | `cycle2_modelling.ipynb` | Baseline model ROC |
| `cycle3_roc_baseline.png` | `cycle3_modelling*.ipynb` | Baseline model ROC (C3) |

---

## 5. Datasets

### 5.1 Premier League Matches (Cycle 1)

| Property | Detail |
|----------|--------|
| File | `data/raw/premier_league_matches.csv` |
| Coverage | Premier League seasons 2000/01 – 2017/18 |
| Rows | 6,840 matches |
| Key columns (raw) | `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `Date`, form columns, stats |
| Processed shape | 6,840 rows × 35 columns |
| Features used | 33 (team form, points, streaks, goal difference, team IDs) |
| Target | `FTR`: 0=Away Win, 1=Draw, 2=Home Win |

**Why this dataset?** It covers 18 complete Premier League seasons with consistent feature engineering applied throughout. The large temporal span gives the model diverse team quality distributions to learn from.

### 5.2 Wyscout Shot Events (Cycle 2)

| Property | Detail |
|----------|--------|
| File | `data/raw/events_England.json` |
| Coverage | England (Premier League) 2017/18 |
| Raw events | ~300,000 events of all types |
| After filtering to shots | 8,451 open-play shots |
| Goal rate | 10.8% (912 goals) |
| Player quality source | `data/raw/playerank.json` |
| Features | 9 (location, geometry, body part, half, player rank) |
| Target | `Goal`: 1=goal, 0=no goal |

**Why Wyscout?** Wyscout provides free open-data with rich event tags, including player IDs (for joining player quality scores) and body-part tags. The England 2017/18 season is the most complete single-season dataset available in the Wyscout open dataset.

### 5.3 Player Injuries (Cycle 3)

| Property | Detail |
|----------|--------|
| File | `data/raw/player_injuries.csv` |
| Coverage | ~1,300 player-seasons, multiple years |
| Processed rows | 1,301 player-seasons |
| High Injury rate | 70.2% |
| Features | 17 (physical attributes, FIFA ratings, injury history) |
| Target | `High_Injury`: 1 if ≥28 days injured, else 0 |
| Key date field | `start_year` (used for chronological split) |

**Why this dataset?** It combines physical/physiological attributes (height, weight, BMI) with FIFA performance ratings and historical injury data — a richer feature set than location-only injury datasets. The multi-season span allows chronological evaluation.

---

## 6. Machine Learning Pipeline (Per Cycle)

Each cycle follows the same six-stage process:

### Stage 1: Exploration
- Load raw data
- Understand schema and data types
- Identify missing values and anomalies
- Identify potential leakage columns
- Visualise target variable distribution and class balance

### Stage 2: Preprocessing
- Drop leakage columns
- Encode categorical variables
- Derive new features from existing ones
- Impute missing values
- Write processed CSV to `data/processed/`

### Stage 3: Modelling (Random Split Baseline)
- Random 80/20 split (stratified for binary tasks)
- Train 5 models: Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluate with appropriate metric (accuracy for Cycle 1; AUC-ROC for Cycles 2 & 3)
- Establish baseline floors and ceilings

### Stage 4: Tuning
- `RandomizedSearchCV` (50 combinations × 5-fold stratified CV)
- Grid covers model-specific hyperparameters (depth, learning rate, regularisation, class weight)
- Save best model + scaler + feature list to `models/`

### Stage 5: Chronological Evaluation (Cycles 1 & 3)
- Sort data by time (Season for Cycle 1; start_year for Cycle 3)
- First 80% = training, last 20% = test
- Repeat Stages 3 and 4 on the chronological split
- Deployed model comes from chronological tuning

### Stage 6: Explainability
- Load deployed model
- Rebuild test set
- Compute SHAP values with `shap.TreeExplainer`
- Generate global importance, beeswarm, and waterfall plots
- Save plots to `docs/`

---

## 7. Model Artefacts and Persistence

### Why joblib instead of pickle?
`joblib` is optimised for large NumPy arrays (which sklearn/XGBoost/LightGBM models contain internally). It is faster and more memory-efficient than Python's native `pickle` for ML objects.

### Loading at startup
The API's `lifespan` function loads all models once at startup:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_registry()          # loads all .pkl files into memory
    build_feature_store()    # builds rolling team stats cache
    yield
```

This means:
- Zero disk I/O at prediction time (no re-loading models per request)
- Sub-millisecond inference latency once running
- A `/health/ready` endpoint confirms startup completion

### Three-artefact pattern
Every cycle stores three files:
```
model.pkl        — the trained estimator
scaler.pkl       — the fitted StandardScaler (Cycles 2 & 3 only)
feature_cols.pkl — ordered list of feature names
```

At inference time, inputs are built as a DataFrame in the exact column order specified by `feature_cols.pkl`, scaled with `scaler.pkl`, then passed to `model.pkl`.

---

## 8. FastAPI Backend

### Application structure

`api/main.py` uses FastAPI's `lifespan` context manager to load models at startup and shut down cleanly:

```python
app = FastAPI(title="Football Predictor API", version="2.0.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(models.router)
app.include_router(teams.router)
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health/ready` | Returns 200 when models are loaded |
| POST | `/predict` | Run any of the three models (or pipeline mode) |
| POST | `/explain` | Compute SHAP values for any prediction |
| GET | `/models` | List all loaded models with metadata |
| GET | `/models/{name}` | Detailed model info and performance comparison |
| GET | `/teams` | List all teams in the feature store |
| GET | `/teams/{team_id}` | Get a specific team's latest stats |

### Request/Response design (POST /predict)

The `PredictionRequest` schema uses a single `model` field as a discriminator:

```json
{
  "model": "match",
  "home_team_id": 5,
  "away_team_id": 12,
  "mw": 20
}
```

```json
{
  "model": "xg",
  "X": 85.0,
  "Y": 50.0,
  "right_foot": 1,
  "left_foot": 0,
  "header": 0,
  "first_half": 1,
  "player_rank": 7.5
}
```

```json
{
  "model": "injury",
  "age": 28,
  "height_cm": 181,
  "weight_kg": 76,
  ...
}
```

### Inference-time feature engineering (`api/ml/pipeline.py`)

Three functions handle the feature construction at inference:

**`build_match_vector(home_team_id, away_team_id, feature_cols, mw, ...)`**
- Looks up each team's latest stats from the feature store (or uses overrides)
- Assembles the 33-column DataFrame
- No scaling applied (XGBoost tree model, not sensitive to scale)

**`build_xg_vector(X, Y, left_foot, right_foot, header, first_half, player_rank, feature_cols)`**
- Converts Wyscout 0–100 coordinates to metres using pitch dimensions (105m × 68m)
- Computes `Distance` (Euclidean to goal centre) and `Angle` (subtended by goalposts)
- Returns DataFrame + distance_m + angle_deg (for display)

**`build_injury_vector(data, feature_cols)`**
- Pass-through: all 17 features are provided directly by the user
- Selects and orders columns to match training order

### Feature store (`api/ml/feature_store.py`)

Loaded at startup from `premier_league_matches_processed.csv`. For every team that appears in the dataset, the feature store captures their latest-known season-to-date snapshot (highest `season × 100 + mw` row). This includes:
- Goals scored/conceded, points, goal difference
- Win/loss streaks (3-match and 5-match)
- Last 5 match results (encoded)
- Form points (sum of last 5 results)

When a user requests a match prediction with `model="match"`, the API looks up both teams' stats automatically — the user only needs to provide team IDs.

### Team ID encoding

Team IDs are derived from alphabetical sorting of all team names in the raw dataset:
```python
all_teams = sorted(pd.concat([raw["HomeTeam"], raw["AwayTeam"]]).unique())
team_map = {team: idx for idx, team in enumerate(all_teams)}
```
This produces deterministic, reproducible IDs consistent between the preprocessing notebook and the API.

---

## 9. Streamlit Dashboard

### Application structure

The dashboard uses Streamlit's multi-page app format:
```
dashboard/
├── app.py                 (landing page — Streamlit runs this)
└── pages/
    ├── 1_Match_Predictor.py
    ├── 2_xG_Calculator.py
    └── 3_Injury_Risk.py
```

The numeric prefix controls page order in the sidebar navigation.

### Page 1: Match Outcome Predictor

**Features:**
- Dropdown selectors for home/away teams (populated from `GET /teams`)
- Three stats input modes:
  - **Auto:** Loads latest team stats from the feature store automatically
  - **Manual:** Pre-fills with stored stats but allows user edits
  - **CSV upload:** User uploads a CSV with custom home/away stats
- Matchweek input
- Result displayed as: outcome label, confidence (high/medium/low), probability bar chart (Plotly)
- Optional SHAP feature explanation bar chart

### Page 2: xG Calculator

**Features:**
- X/Y sliders for shot location (0–100 Wyscout scale)
- Foot selection (Right/Left/Header)
- First half toggle
- Player rank slider
- Result displayed as: xG value (large coloured number), distance to goal, angle, prediction
- Interactive pitch visualisation (Plotly) showing shot location
- Optional SHAP bar chart

### Page 3: Injury Risk Predictor

**Features:**
- Three-column input form covering physical attributes, demographics, and injury history
- Separate inputs for career cumulatives vs. previous-season stats
- BMI auto-computed but also manually editable
- Result displayed as: injury risk gauge (Plotly), High/Low Injury Risk label, probabilities
- Optional SHAP bar chart

### API client (`dashboard/components/api_client.py`)

All HTTP calls go through a single module. This isolates the API URL (default: `http://localhost:8000`) from the dashboard logic. The `is_api_ready()` function checks `/health/ready` and drives the sidebar status indicator.

### Charts (`dashboard/components/charts.py`)

Reusable Plotly chart builders:
- `match_probability_chart()` — Grouped bar chart of Home Win / Draw / Away Win probabilities
- `xg_pitch()` — Pitch diagram with shot marker sized/coloured by xG
- `injury_gauge()` — Semi-circular gauge showing injury risk probability
- `shap_bar_chart()` — Horizontal bar chart of SHAP feature contributions

---

## 10. The Unified Pipeline Mode

Pipeline mode (`model="pipeline"`) is the key integration feature that distinguishes this project from three isolated models.

### How it works

```
POST /predict  {model: "pipeline", home_team_id: 5, away_team_id: 12,
                home_shots: [...], away_shots: [...],
                home_players: [...], away_players: [...]}
```

**Step 1 — Base match prediction (Cycle 1)**
- Match model produces base probabilities: {home_win: P1, draw: P2, away_win: P3}
- These reflect historical team statistics only

**Step 2 — xG aggregation (Cycle 2)**
- The xG model runs on every shot in `home_shots` and `away_shots`
- Aggregates: `total_xg`, `avg_xg_per_shot`, `shot_count` per team

**Step 3 — Injury aggregation (Cycle 3)**
- The injury model runs on every player in `home_players` and `away_players`
- Aggregates: `high_risk_count`, `avg_risk_probability` per team

**Step 4 — Probability adjustment**
```python
# xG adjustment: better chance quality → nudge probabilities
if home_avg_xg > away_avg_xg:
    p_home += min(0.05, (home_avg - away_avg) * 0.5)
    p_away -= adjustment

# Injury adjustment: more high-risk players → disadvantage
net = away_high_risk - home_high_risk
p_home += min(0.05, net * 0.02)

# Re-normalise to sum to 1.0
```

Adjustments are capped at ±5% per factor to ensure the match model's learned probabilities always dominate. The pipeline is designed to *refine* the match prediction, not override it.

**Step 5 — Response**
The response includes:
- `prediction` and `probabilities` (adjusted)
- `base_probabilities` (from match model alone)
- `xg_analysis` (per team)
- `injury_analysis` (per team)
- `pipeline_factors` — plain-English explanation of what adjustments were made and why

### Design rationale

The pipeline mode is designed around the principle that *no single model has the full picture*. The match model knows about historical team quality; the xG model knows about chance quality in this specific match; the injury model knows about squad fitness. Combining them produces a richer prediction than any model alone.

---

## 11. SHAP Explainability System

SHAP is used in two contexts:

### Offline (notebook explainability)
Each cycle has a dedicated `cycle_X_explainability.ipynb` that:
1. Loads the deployed model
2. Rebuilds the test set
3. Computes SHAP values with `shap.TreeExplainer`
4. Produces global importance, beeswarm, and waterfall plots saved to `docs/`

These plots are used for the report and provide a comprehensive global understanding of what the models have learned.

### Online (API explainability)
The `/explain` endpoint computes SHAP values at inference time for any individual prediction. This enables the Streamlit dashboard to show per-prediction feature attributions.

```
POST /explain  {model: "match", home_team_id: 5, away_team_id: 12, top_n: 10}
```

Response includes `top_features` (list of feature/value/shap_value/impact) and `all_features` (full list).

### Why TreeExplainer?

`shap.TreeExplainer` is the exact SHAP algorithm for tree-based models. It:
- Is algorithmically exact (not an approximation)
- Exploits the tree structure for O(TLD) computation (T=trees, L=leaves, D=depth)
- Returns consistent SHAP values that sum to the model output minus the baseline
- Is far faster than KernelExplainer for tree models

### SHAP interpretation guide

| SHAP value | Meaning |
|-----------|---------|
| Positive | This feature pushed the prediction toward the positive class |
| Negative | This feature pushed the prediction away from the positive class |
| Near zero | This feature had minimal influence on this prediction |
| Large magnitude | This feature was decisive for this prediction |

The `base_value` returned by the API is the model's average output across the training dataset — the starting point before any features are applied.

---

## 12. Configuration System

`config.py` provides a single `Paths` class that all notebooks and API modules import:

```python
class Paths:
    ROOT = Path(__file__).parent
    DATA_RAW       = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"

    PL_MATCHES_RAW            = DATA_RAW / "premier_league_matches.csv"
    WYSCOUT_PROCESSED         = DATA_PROCESSED / "wyscout_shots_processed.csv"
    PLAYER_INJURIES_PROCESSED = DATA_PROCESSED / "player_injuries_processed.csv"

    C1_MODEL    = ROOT / "models" / "cycle1" / "cycle1_xgb_best.pkl"
    C2_MODEL    = ROOT / "models" / "cycle2" / "cycle2_best_model.pkl"
    ...
```

The `ensure_dirs()` function creates all model subdirectories on import, preventing `FileNotFoundError` when a notebook runs for the first time.

**Why a shared config?** Before this was introduced, notebooks and API modules used ad-hoc relative paths (`../../data/raw/...`) that broke depending on where the file was run from. The shared config resolves paths from the project root reliably.

---

## 13. Design Decisions and Rationale

### 13.1 Single `/predict` endpoint vs. separate endpoints

All predictions go through `POST /predict` with a `model` discriminator. This was chosen over three separate endpoints (`/predict/match`, `/predict/xg`, `/predict/injury`) because:
- A single endpoint simplifies client code (only one URL to call)
- The pipeline mode naturally fits within the same structure
- Validation can share common fields (team IDs, optional overrides)

### 13.2 Chronological split for Cycles 1 and 3

**Cycle 1:** Match outcome has temporal structure — team quality, tactics, and league competitiveness change between 2000 and 2018. A random split leaks later seasons into the training set. Chronological split faithfully simulates deployment.

**Cycle 3:** Injury patterns may shift with changes in training science and medical protocols. Chronological split tests whether 2016–2019 patterns generalise to the 2020 season.

**Cycle 2:** Shot data has no temporal ordering requiring preservation. Each shot is an independent event; the 2017/18 season provides a self-contained dataset. Random stratified split is appropriate.

### 13.3 AUC-ROC for binary classification (Cycles 2 & 3)

Both Cycle 2 and Cycle 3 have significant class imbalance:
- Cycle 2: 10.8% goal rate (9:1 imbalance)
- Cycle 3: 70.2% high injury rate (3:1 imbalance)

AUC-ROC measures the model's ability to *rank* positive cases above negative cases across all classification thresholds. It is threshold-independent and immune to class imbalance effects that distort accuracy.

### 13.4 Three separate models vs. a unified model

Each cycle uses a fundamentally different feature set and target variable. Trying to build a single model would require either:
- Joining all datasets (impossible — different granularities and time periods)
- Using only common features (losing most information from each dataset)

Three separate specialised models is the correct architecture.

### 13.5 No neural networks

All three cycles use gradient boosting (XGBoost/LightGBM) rather than neural networks because:
- All three datasets are tabular (structured data)
- Dataset sizes are small-to-medium (1,301 to 8,451 rows)
- Neural networks generally underperform gradient boosting on small tabular datasets
- Gradient boosting provides native SHAP support with exact computation
- Training is fast (seconds) and requires no GPU

### 13.6 Pipeline adjustment caps at ±5%

The xG and injury adjustments to match probabilities are capped at ±5% each. This prevents the pipeline from producing nonsensical results (e.g., a team with one good shot chance getting a 90% win probability). The match model's historical patterns are the most reliable signal; xG and injury refine rather than override.

---

## 14. Limitations

### Data limitations
1. **Cycle 1 ends at 2018** — The model does not know about recent Premier League dynamics, current clubs, or modern playing styles.
2. **Cycle 2 is single-season** — One season (~8,451 shots) is a relatively small sample for an xG model. Commercial models use hundreds of thousands of shots across multiple seasons and leagues.
3. **Cycle 3 has 1,301 rows** — Small dataset for complex ML. Many feature combinations are underrepresented.
4. **No real-time data** — The feature store is static (loaded from the processed CSV). A production system would update team stats after each match.

### Model limitations
1. **Draw prediction** (Cycle 1) — The model has very low recall for draws. This is a known limitation of aggregate-statistics-based match prediction models.
2. **Injury causality** (Cycle 3) — The model identifies correlates of injury, not causes. It cannot reason about the actual mechanisms (overtraining, contact injuries, pitch conditions).
3. **xG independence assumption** (Cycle 2) — The model treats each shot as independent. In reality, match state, fatigue, and defensive organisation affect shot quality.

### System limitations
1. **No authentication** — The API has no authentication. For production deployment, API keys or OAuth would be required.
2. **No database** — Predictions are not stored. A production system would persist predictions for analytics.
3. **Single-machine deployment** — The system runs on a single server; no load balancing or horizontal scaling is implemented.

---

## 15. Future Improvements

### Near-term
1. **Live data feeds** — Connect the feature store to a live Premier League data API (e.g., football-data.org) to update team stats automatically after each match.
2. **More shot data** — Add Wyscout data from other seasons and leagues (Serie A, Bundesliga, La Liga) to improve xG model generalisation.
3. **Better injury features** — Incorporate GPS tracking data or training load metrics where available.

### Medium-term
4. **Poisson goal model** — Implement a goals-based match prediction model (Dixon-Coles or similar) alongside the classification model for better draw prediction.
5. **Time-series features** — Add momentum features (e.g., goals per match in the last 3 games weighted by recency) for Cycle 1.
6. **Model retraining pipeline** — Automate retraining when new match/shot data is available, with automatic performance regression checks.

### Long-term
7. **Player-level match prediction** — Extend Cycle 1 to incorporate lineup-level features (which players are starting, their injury risk from Cycle 3).
8. **Deep learning exploration** — For xG, experiment with graph neural networks on spatial shot data or LSTM on sequential match event data.
9. **Production infrastructure** — Containerise with Docker, add a PostgreSQL prediction store, implement proper authentication and rate limiting.

---

*Documentation generated from project state as of May 2026. All performance metrics and results are directly from notebook outputs.*
