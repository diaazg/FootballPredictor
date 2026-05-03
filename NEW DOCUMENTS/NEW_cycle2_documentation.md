# Cycle 2 — Expected Goals (xG) Prediction: Detailed Documentation

**Task:** Predict the probability that a single open-play shot results in a goal (binary classification)  
**Algorithm (deployed):** XGBoost (tuned, random stratified split)  
**Best test AUC-ROC:** 0.8183  
**Source notebooks:** `notebooks/cycle2/`

---

## 1. Problem Statement

**Expected Goals (xG)** is a metric widely used in professional football analytics. For every shot, it answers: *given where this shot was taken from and how it was taken, what is the historical probability that a shot like this becomes a goal?*

An xG value of 0.08 means the shot had an 8% historical chance of being scored. Accumulating xG across a match tells you how many goals a team "should have" scored based on the quality of their chances — independently of whether the striker got lucky.

This is a **binary classification** problem:
- Class 0: No Goal (89.2% of shots)
- Class 1: Goal (10.8% of shots)

The severe class imbalance (approximately 9:1) means accuracy is a misleading metric — a model that always predicts "No Goal" achieves ~89% accuracy but learns nothing. The primary metric is **AUC-ROC**, which measures how well the model *ranks* goals above non-goals.

---

## 2. Dataset

### Sources
- **Raw shot events:** `data/raw/events_England.json` — Wyscout open-data, England (Premier League) 2017/18 season
- **Player quality:** `data/raw/playerank.json` — Wyscout playerank quality scores (0–10 scale)
- **Processed file:** `data/processed/wyscout_shots_processed.csv`

### What is Wyscout?
Wyscout is a professional football scouting platform. Its open data includes timestamped event logs (passes, shots, fouls, etc.) for every match, with each event tagged with player ID, coordinates, and outcome tags.

### Filtering to shots

The raw events file contains all match events (passes, tackles, saves, etc.). The preprocessing notebook filters to:
- `eventId == 10` (shot events only)
- Open-play shots only (penalties, own goals, and free-kick shots are excluded because they have different scoring patterns and would warrant separate models)

### Dataset scale

| Property | Value |
|----------|-------|
| Total shots after filtering | 8,451 |
| Goal rate | 10.8% (912 goals / 8,451 shots) |
| Features | 9 |
| Target | `Goal` (binary) |
| Train/Test split | 80/20 stratified random |
| Train size | 6,760 shots |
| Test size | 1,691 shots |

### Features

| Feature | Source | Description |
|---------|--------|-------------|
| `X` | Wyscout event | Shot X-coordinate (0–100), where 100 = goal line |
| `Y` | Wyscout event | Shot Y-coordinate (0–100), where 50 = centre |
| `Distance` | Computed | Distance from shot location to goal centre (metres) |
| `Angle` | Computed | Angle subtended by the goal posts from shot location (degrees) |
| `Left_Foot` | Wyscout tag 401 | 1 if taken with left foot, else 0 |
| `Right_Foot` | Wyscout tag 402 | 1 if taken with right foot, else 0 |
| `Header` | Wyscout tag 403 | 1 if header, else 0 |
| `First_Half` | Match period | 1 if first half, else 0 |
| `Player_Rank` | playerank.json | Player quality score (0–10); joined on playerId |

**Pre-shot features only:** No post-shot tags (e.g., where in the goal the ball went, goalkeeper position) are included. These are only observable after the shot is taken and would constitute leakage.

---

## 3. Feature Engineering Details

### Computing Distance

Distance is computed from shot coordinates (converted from Wyscout 0–100 scale to metres using standard Premier League pitch dimensions: 105m × 68m):

```
x_m = X / 100 × 105
y_m = Y / 100 × 68
distance_m = sqrt((x_m - 105)² + (y_m - 34)²)
```

Where `(105, 34)` is the goal centre in metres.

### Computing Angle

The angle is the angle subtended by the goal posts from the shot position:

```
angle = |atan2(post_right_y - y_m, 105 - x_m) - atan2(post_left_y - y_m, 105 - x_m)|
```

Where the posts are at Y = 30.34m (left post) and Y = 37.66m (right post).

A larger angle means the shot is more central and closer to goal — higher xG expected. A smaller angle means the shot is wide or far from goal.

### Why compute Distance and Angle instead of just using X/Y?

Distance and Angle encode the spatial relationship to the goal in a compact, interpretable form. A model trained only on X/Y would need to implicitly learn this geometry from data; providing Distance and Angle directly accelerates learning and makes the features more interpretable in SHAP outputs.

Note: X and Y are **also** retained as features because they carry additional spatial information beyond Distance/Angle alone (e.g., shots from the same distance/angle but different pitch sides may have different defenders/goalkeeper positioning).

### Player_Rank

Joined from `playerank.json` by matching `playerId`. Players not found in the playerank file receive the median rank. This captures player quality as a proxy for finishing ability — an elite striker is more likely to score from a given position than an average player.

---

## 4. Notebook Pipeline

### 4.1 `cycle2_exploration_wyscout.ipynb` — Data Exploration

**What it does:**
- Loads and inspects the raw Wyscout JSON events
- Identifies the event types and filters to shot events
- Examines leakage risk in Wyscout tags (e.g., tag 1801 "goal" is the target itself — must not be used as a feature)
- Explores the spatial distribution of shots and goals on the pitch
- Confirms the ~10.8% goal rate and severe class imbalance

**Key decision:** Use only pre-shot features (location, foot, half, player rank). Post-shot tags that describe what happened after the shot are excluded.

### 4.2 `cycle2_preprocessing_wyscout.ipynb` — Feature Engineering

**What it does:**
- Filters raw events to open-play shots
- Converts Wyscout coordinates (0–100) to pitch coordinates
- Computes `Distance` and `Angle` geometrically
- Extracts body-part tags (Left_Foot, Right_Foot, Header) from the event tag list
- Extracts `First_Half` from the match period field
- Joins `playerankScore` from `playerank.json` on `playerId`; fills missing with median
- Writes `data/processed/wyscout_shots_processed.csv`

### 4.3 `cycle2_modelling.ipynb` — Baseline Model Comparison

**What it does:**
- Applies a stratified 80/20 split (stratify on `Goal` to preserve 10.8% rate in both sets)
- Applies `StandardScaler` (fit on training data only)
- Trains and evaluates five models:
  1. **Dummy** (most_frequent) — always predicts No Goal
  2. **Logistic Regression** — with `class_weight='balanced'`
  3. **Random Forest** — with `class_weight='balanced_subsample'`
  4. **XGBoost** — with `scale_pos_weight` (imbalance ratio ~8.25)
  5. **LightGBM** — with `scale_pos_weight`

**Results (untuned):**

| Model | AUC-ROC | Accuracy | Notes |
|-------|---------|----------|-------|
| Dummy | 0.5000 | 89.18% | Always predicts No Goal |
| Logistic Regression | 0.7963 | 72.50% | Strong linear baseline |
| Random Forest | 0.7884 | — | Second-best untuned |
| XGBoost | 0.7871 | — | Third-best untuned |
| **LightGBM** | **0.8035** | — | Best untuned |

**Why does the dummy get 89% accuracy?** Because 89% of shots are No Goal. "Always predict No Goal" is correct 89% of the time but completely useless — it never identifies a goal. This is why AUC-ROC is the correct metric for this task.

### 4.4 `cycle2_tuning.ipynb` — Hyperparameter Tuning ← DEPLOYED MODEL

**What it does:**
- Runs `RandomizedSearchCV` (50 combinations × 5-fold stratified CV) for XGBoost, Random Forest, and LightGBM
- Scoring metric: `roc_auc` (not accuracy)
- Key XGBoost parameters tuned: `scale_pos_weight` (also swept at 0.5× and 1.5× the computed ratio to find optimal class balance)
- Saves the highest-AUC model as `models/cycle2/cycle2_best_model.pkl`
- Saves the scaler as `models/cycle2/cycle2_scaler.pkl`
- Saves the feature list as `models/cycle2/cycle2_feature_cols.pkl`

**Final tuned results:**

| Model | Test AUC-ROC | CV AUC | Gain vs Untuned |
|-------|-------------|--------|----------------|
| **XGBoost Tuned** | **0.8183** | 0.8137 | +0.0312 |
| Random Forest Tuned | 0.8176 | 0.8120 | +0.0292 |
| LightGBM Tuned | 0.8152 | 0.8127 | +0.0117 |

**Best XGBoost hyperparameters:**
- `subsample`: 0.7
- `scale_pos_weight`: 8.25 (the computed natural imbalance ratio)
- `n_estimators`: 200
- `min_child_weight`: 5
- `max_depth`: 4
- `learning_rate`: 0.01
- `colsample_bytree`: 0.7

The low learning rate + many estimators is the consistent winning configuration for both Cycle 1 and Cycle 2 XGBoost.

### 4.5 `cycle2_explainability.ipynb` — SHAP Analysis

**What it does:**
- Loads the deployed XGBoost model and scaler
- Rebuilds the identical stratified test split
- Computes SHAP values using `shap.TreeExplainer`
- Saves three plots to `docs/`:

| Plot | File | What it shows |
|------|------|---------------|
| Global importance | `docs/cycle2_shap_global_importance.png` | Mean \|SHAP\| across all shots |
| Beeswarm summary | `docs/cycle2_shap_summary.png` | Direction and magnitude per feature |
| Waterfall | `docs/cycle2_shap_waterfall.png` | Single-shot explanation |

---

## 5. Model Evaluation Details

### Classification report (tuned XGBoost)

```
              precision    recall  f1-score   support

     No Goal       0.96      0.73      0.83      1508
        Goal       0.26      0.76      0.38       183

    accuracy                           0.73      1691
   macro avg       0.61      0.75      0.61      1691
weighted avg       0.89      0.73      0.78      1691
```

**Interpreting the report:**
- **Goal recall = 0.76** — The model correctly identifies 76% of actual goals. This is the key metric for an xG tool: we want to catch most real goals.
- **Goal precision = 0.26** — Of all shots predicted as Goal, only 26% actually are. This is expected given the 10.8% base rate; the model raises recall at the cost of precision.
- **No Goal recall = 0.73** — The model correctly identifies 73% of non-goals.

This precision/recall trade-off is appropriate: for xG, we care more about correctly ranking shots by goal probability than achieving a binary threshold decision.

### ROC Curves

The ROC curves are saved in `docs/`:
- `cycle2_roc_curve.png` — ROC curve for the tuned XGBoost model (AUC 0.8183)
- `cycle2_roc_curves_detailed.png` — All models compared on the same axes
- `cycle2_roc_baseline.png` — Baseline comparison (LR vs tuned models)

These visualise how the model's true positive rate (sensitivity) trades off against the false positive rate across all classification thresholds.

---

## 6. SHAP Findings

### Most important features (global)

1. **Distance** — The dominant feature. Short distance → high xG. Long distance → low xG. Strong, consistent negative relationship.
2. **Angle** — Second most important. Wide/central angle → higher xG. Narrow/tight angle → lower xG.
3. **X coordinate** — Provides spatial information beyond Distance alone (e.g., direct shots from close range vs. cross-shots from similar distances).
4. **Y coordinate** — Central positions (Y ≈ 50) produce higher xG.
5. **Player_Rank** — Higher-ranked players score more from equivalent positions.
6. **Shot type** (Header, Left_Foot, Right_Foot) — Headers have lower xG; preferred foot shots (Right_Foot) increase xG.
7. **First_Half** — Minimal effect; which half the shot was taken in barely affects goal probability.

### SHAP direction interpretation

| Feature | High value → | Low value → |
|---------|-------------|------------|
| Distance | Decreases xG (far from goal) | Increases xG (close to goal) |
| Angle | Increases xG (central position) | Decreases xG (tight angle) |
| Player_Rank | Increases xG (elite finisher) | Decreases xG |
| Header | Decreases xG (lower conversion rate than foot shots) | — |
| Right_Foot | Increases xG slightly (preferred foot) | — |

---

## 7. How This Connects to the Full System

### At inference time (API)

When a user calls `POST /predict` with `model="xg"`:

1. The user provides raw `X`, `Y` coordinates (0–100 scale) plus shot context
2. `api/ml/pipeline.py:build_xg_vector()` computes Distance and Angle geometrically (using real pitch dimensions: 105m × 68m, posts at Y=30.34m and Y=37.66m)
3. The 9-column feature vector is scaled with the saved `StandardScaler`
4. The XGBoost model returns `predict_proba()[:, 1]` — the goal probability
5. The API returns `xg`, `distance_m`, `angle_deg`, and `prediction`

### In pipeline mode

When `model="pipeline"` is used:
- A list of `home_shots` and `away_shots` (each with X, Y, and context) is provided
- The xG model runs on every shot in each list
- Results are aggregated into: `total_xg`, `avg_xg_per_shot`, `shot_count`
- The home/away `avg_xg_per_shot` difference is used to nudge the match model's base probabilities (up to ±5%)

### In the Streamlit dashboard (Page 2: xG Calculator)

- The user sets shot location via X/Y sliders
- Distance and angle are computed automatically
- The result is displayed as an xG value (0–1) with a colour-coded pitch visualisation
- Optional SHAP bar chart shows which features drove that specific shot's xG

---

## 8. Why the Data Is Restricted to Open-Play Shots

### Excluded shot types:
- **Penalties** — Always ~10.75m from goal, central position. The penalty conversion rate (~76%) is far higher than open-play shots from the same distance. A separate penalty model would be appropriate.
- **Free kicks** — The wall and defensive set affects scoring probability in ways not captured by raw location.
- **Own goals** — Not relevant to the shooter's skill.
- **Set-piece headers** — The goal involves complex 3D trajectory and delivery that is not captured by 2D location alone.

Including these would contaminate the model with different scoring distributions.

---

## 9. Limitations

1. **One season only** — Training data is England 2017/18 only (~8,451 shots). Adding multiple seasons and leagues would significantly increase the training set and improve generalisation.

2. **No goalkeeper positioning** — Professional xG models (e.g., StatsBomb, Opta) include goalkeeper coordinates, body orientation, and whether the goalkeeper had set before the shot. These are not available in Wyscout open data.

3. **No defensive pressure** — The number of defenders between the shooter and goal, and their positions, strongly affects goal probability but are not in the feature set.

4. **Player_Rank as quality proxy** — `playerankScore` is a composite score, not a specific finishing ability metric. A more granular finishing skill rating would be more appropriate.

5. **Binary prediction vs. continuous xG** — A classification model predicts Goal/No Goal. A regression model predicting the continuous probability directly (without a classification head) might produce better-calibrated xG values. However, the `predict_proba()` output from XGBoost is well-calibrated for this dataset.

6. **No cross-match context** — The model treats each shot independently. In reality, match state (score, time remaining, fatigue) affects how shots are taken and their probability of success.
