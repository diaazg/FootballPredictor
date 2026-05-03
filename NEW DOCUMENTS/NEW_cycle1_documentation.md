# Cycle 1 — Match Outcome Prediction: Detailed Documentation

**Task:** Predict the full-time result of a Premier League match: **Home Win**, **Draw**, or **Away Win**  
**Algorithm (deployed):** XGBoost (tuned, chronological split)  
**Best test accuracy:** 52.85% (vs 46.35% dummy baseline)  
**Source notebooks:** `notebooks/cycle1/`

---

## 1. Problem Statement

Given a set of pre-match statistics about both teams (season-to-date performance, recent form, goal difference, win/loss streaks), can we predict whether the home team wins, the match ends in a draw, or the away team wins?

This is a three-class classification problem with an inherent imbalance: historically, home teams win roughly 46% of matches, draws occur ~26% of the time, and away teams win ~28%.

### Why is this hard?

Football is notoriously difficult to predict. The noise-to-signal ratio is very high: a top team can lose to a bottom team, goals can come from errors or penalties, and a single red card can change the match entirely. The theoretical ceiling for a purely statistical model is generally considered to be around 55–60% — anything above 52% on clean data is meaningful.

---

## 2. Dataset

### Source
- **Raw file:** `data/raw/premier_league_matches.csv`
- **Processed file:** `data/processed/premier_league_matches_processed.csv`

### Scale
- **6,840 matches** across 18 Premier League seasons (2000/01 to 2017/18)
- **35 columns** in the processed dataset (33 features + target + Season)

### Processed Features (33)

| Category | Features |
|----------|----------|
| Team Goals (season-to-date) | `HTGS`, `ATGS` (goals scored), `HTGC`, `ATGC` (goals conceded) |
| Points | `HTP`, `ATP` (cumulative points) |
| Recent form (last 5) | `HM1–HM5`, `AM1–AM5` (W=3, D=1, L=0, Missed=0) |
| Form summary | `HTFormPts`, `ATFormPts` (sum of last 5 form scores) |
| Win streaks | `HTWinStreak3`, `HTWinStreak5`, `ATWinStreak3`, `ATWinStreak5` |
| Loss streaks | `HTLossStreak3`, `HTLossStreak5`, `ATLossStreak3`, `ATLossStreak5` |
| Goal difference | `HTGD`, `ATGD`, `DiffPts`, `DiffFormPts` |
| Match context | `MW` (matchweek, 1–38) |
| Team identity | `HomeTeam`, `AwayTeam` (label encoded, 0–N) |

**Target:** `FTR` (Full Time Result) — 0 = Away Win, 1 = Draw, 2 = Home Win

### Class distribution

| Class | Count | Proportion |
|-------|-------|-----------|
| Home Win (2) | ~3,170 | ~46.3% |
| Away Win (0) | ~1,921 | ~28.1% |
| Draw (1) | ~1,749 | ~25.6% |

---

## 3. Notebook Pipeline

The cycle follows six notebooks in order:

### 3.1 `cycle1_exploration.ipynb` — Data Exploration

**What it does:**
- Loads and inspects the raw Premier League CSV
- Examines the distribution of FTR (target variable)
- Identifies potential data leakage (e.g., full-time score columns `FTHG`, `FTAG` directly reveal the target and must be dropped)
- Checks for missing values and scale issues in numerical features
- Explores the correlation between features and match outcomes

**Key finding:** `FTHG` and `FTAG` (full-time home/away goals) are leakage features — they determine `FTR` directly and must be excluded.

### 3.2 `cycle1_preprocessing.ipynb` — Feature Engineering

**What it does:**
- Reconstructs `FTR` from `FTHG`/`FTAG` as a numeric label (0/1/2), then drops the goal columns
- Encodes form columns (`HM1–HM5`, `AM1–AM5`): W→3, D→1, L→0, Missing→0
- Label-encodes `HomeTeam`/`AwayTeam` alphabetically (each team gets a unique integer ID)
- Extracts `Season` (year) from the `Date` column, drops `Date`
- Writes the cleaned, feature-engineered CSV to `data/processed/`

**Why encode form as W=3/D=1/L=0?** This preserves the ordinal relationship between results (win is worth 3 points, draw 1, loss 0), consistent with the actual Premier League points system.

### 3.3 `cycle1_modelling.ipynb` — Baseline Model Comparison (Random Split)

**What it does:**
- Applies a random 80/20 train/test split
- Trains and evaluates five models:
  1. **Dummy** (most_frequent) — always predicts Home Win
  2. **Logistic Regression** — linear baseline
  3. **Random Forest** — ensemble of decision trees
  4. **XGBoost** — gradient boosted trees
  5. **LightGBM** — fast gradient boosting

**Results (random split, untuned):**

| Model | Test Accuracy |
|-------|--------------|
| Dummy (most_frequent) | 46.35% |
| Logistic Regression | ~49.00% |
| Random Forest | ~51.39% |
| XGBoost | ~50.00% |
| LightGBM | ~49.50% |

**Key observation:** Even the simple Random Forest beats the dummy baseline by ~5 percentage points, confirming that match statistics contain predictive signal.

### 3.4 `cycle1_tuning.ipynb` — Hyperparameter Tuning (Random Split)

**What it does:**
- Runs `RandomizedSearchCV` (50 combinations × 5-fold stratified CV) over XGBoost, Random Forest, and LightGBM
- Scoring metric: accuracy (appropriate for roughly balanced 3-class problem after majority correction)
- Searches over: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma

**Result:** Tuned XGBoost achieves 52.78% on the random split. Best XGBoost params: low learning rate (0.01), 200 trees, conservative regularisation.

### 3.5 `chronological/cycle1_modelling_chronological.ipynb` — Baseline (Chronological Split)

**What it does:**
- Sorts all matches by `Season` (chronological order)
- Takes the first 80% of matches as training, last 20% as test
- Trains the same five models on this temporally honest split

**Why chronological?** A random split can leak future seasons into the training set — the model "sees" matches from 2015 while being tested on 2010. Chronological split mirrors real deployment: we always predict future matches from past data.

### 3.6 `chronological/cycle1_tuning_chronological.ipynb` — Tuning (Chronological) ← DEPLOYED MODEL

**What it does:**
- Same `RandomizedSearchCV` grid as the random-split tuning
- Uses the chronological 80/20 split
- Saves the best model as `models/cycle1/cycle1_xgb_best.pkl`
- Saves the feature column list as `models/cycle1/cycle1_feature_cols.pkl`

**Deployed model result:** 52.85% test accuracy on the chronological hold-out

**Note:** No scaler is saved because XGBoost is a tree-based model — it does not require feature normalisation. At inference time, the API passes raw (but correctly ordered) feature values directly to the model.

### 3.7 `cycle1_explainability.ipynb` — SHAP Analysis

**What it does:**
- Loads the deployed model from disk
- Rebuilds the chronological test set
- Computes SHAP values using `shap.TreeExplainer` (exact, fast algorithm for tree models)
- Produces four visualisations saved to `docs/`:

**Outputs:**

| Plot | File | What it shows |
|------|------|---------------|
| Global importance | `cycle1_shap_global_importance.png` | Mean \|SHAP\| across all classes — which features matter most overall |
| Per-class beeswarm | `cycle1_shap_summary_side_by_side.png` | SHAP distributions for Home Win, Draw, Away Win side-by-side |
| Per-class importance | `cycle1_shap_per_class.png` | Feature importance bar charts per outcome |
| Waterfall | `cycle1_shap_waterfall.png` | Single-match explanation: how each feature contributed to one prediction |

---

## 4. Model Evaluation

### Final model comparison

| Model | Split | Test Accuracy | Notes |
|-------|-------|--------------|-------|
| **XGBoost Tuned (chronological)** | Chronological | **52.85%** | **Deployed model** |
| XGBoost Tuned (random) | Random | 52.78% | Near-identical — no temporal leakage concern |
| Random Forest (untuned) | Random | 51.39% | Strong untuned baseline |
| XGBoost (untuned) | Random | 50.00% | Pre-tuning |
| Logistic Regression | Random | ~49.00% | Linear baseline |
| Dummy (most_frequent) | — | 46.35% | Always predicts Home Win |

### Confusion matrix insights

The model shows a well-known pattern for football prediction:
- **Home Win recall is high** (~0.67–0.86) — the model reliably identifies most home wins
- **Away Win recall is moderate** (~0.44–0.52) — the model captures some away wins
- **Draw recall is very low** (~0.05–0.20) — draws are notoriously hard to predict

This is not a flaw in the model — it reflects the fundamental uncertainty of draws. No feature reliably separates "evenly matched teams that draw" from "evenly matched teams where one wins by a fluke goal."

---

## 5. SHAP Findings

### Most important features (global, all classes)

1. **DiffPts** — Point difference between teams. The strongest single predictor. When the home team has significantly more points, the model strongly predicts Home Win.
2. **AM1, HM1** — Most recent match result for each team. Recent form carries more signal than older form.
3. **HTGD, ATGD** — Goal difference (proxy for team quality throughout the season).
4. **HTP, ATP** — Season-to-date points totals.
5. **HomeTeam, AwayTeam** — Team identity (label-encoded). The model has learned team-specific tendencies.
6. **HTFormPts, ATFormPts** — 5-match form summary.
7. **Win/Loss streaks** — Near-zero SHAP values; minimal predictive contribution.

### Per-class SHAP patterns

**Home Win:**
- High DiffPts (home team stronger) → strong positive push
- High HM1 (strong recent home form) → increases Home Win probability
- High AM1 (strong away form) → reduces Home Win probability

**Away Win:**
- Low DiffPts (away team stronger) → strong positive push for Away Win
- High AM1 (strong away form) → increases Away Win probability

**Draw:**
- SHAP values are small and clustered near zero for all features
- No feature strongly discriminates draws — this explains the low draw recall

---

## 6. Feature Engineering Details

### Why encode form as W=3/D=1/L=0?

The encoding reflects the actual points system. A win is worth 3× a draw, which captures the ordinal importance correctly without introducing an arbitrary mapping.

### Why label-encode team names alphabetically?

XGBoost can accept integer-encoded categorical features. The alphabetic encoding is deterministic and reproducible. At inference time, the API uses the same encoding derived from the full alphabetic sort of all teams in the raw dataset.

### Why include team identity at all?

Some teams systematically over- or underperform relative to their stats (e.g., a team that historically scores disproportionate goals from set pieces may not show this in rolling goal stats). The team-identity feature captures team-specific fixed effects not captured by performance statistics alone.

---

## 7. How This Connects to the Full System

The Cycle 1 model is the **core match prediction engine**. In the FastAPI:

1. When a user calls `POST /predict` with `model="match"`, the `prediction_service.predict_match()` function:
   - Looks up each team's latest season-to-date stats from the **feature store** (pre-built rolling cache)
   - Assembles the 33-column feature vector via `api/ml/pipeline.py:build_match_vector()`
   - Passes it directly (no scaling) to the loaded XGBoost model
   - Returns probabilities for all three outcomes

2. In **pipeline mode** (`model="pipeline"`), the Cycle 1 match probabilities are the starting point, which the xG and injury models then adjust.

3. The SHAP explanation endpoint (`POST /explain` with `model="match"`) uses `shap.TreeExplainer` on the same model to compute per-feature contributions at inference time.

---

## 8. Limitations

1. **Data recency** — The model is trained on 2000–2018 data. The Premier League has changed significantly (VAR, different competitive landscape). The team-identity features embed 2000–2018 team-specific patterns that may not generalise to current squads.

2. **No in-match data** — Only pre-match aggregate statistics are used. Factors like starting lineups, player injuries, referee assignments, and weather are not modelled.

3. **Draw prediction** — The model systematically under-predicts draws, a known limitation of statistical football prediction. Draws require different modelling approaches (e.g., Poisson goal models).

4. **Team identity as proxy** — Label encoding team names creates ordinal relationships that don't exist (team 0 is not "half" of team 2). A one-hot encoding or embedding would be more principled but adds 40+ sparse columns.

5. **52.85% accuracy ceiling** — This is competitive with published research on Premier League prediction using tabular statistics, but there is an inherent ceiling on how predictable football matches are from aggregate stats alone.
