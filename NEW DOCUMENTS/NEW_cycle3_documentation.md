# Cycle 3 — Player Injury Risk Prediction: Detailed Documentation

**Task:** Predict whether a player will miss 28+ days in the upcoming season (binary classification)  
**Algorithm (deployed):** LightGBM (tuned, chronological split)  
**Best test AUC-ROC:** 0.6800  
**Source notebooks:** `notebooks/cycle3/`

---

## 1. Problem Statement

Injuries are one of the most costly events in professional football. A key player missing several weeks can eliminate a team from a cup, cost points in a title race, or derail a squad's entire season. If a club's medical staff could identify which players are at elevated injury risk *before* the season begins, they could tailor pre-season training load, rehabilitation protocols, or transfer decisions accordingly.

This cycle asks: **given a player's physical attributes, FIFA ratings, demographics, and prior-season injury history, can we predict whether they will miss 28 or more days due to injury in the upcoming season?**

This is a **binary classification** problem:
- Class 0: Low Injury Risk (fewer than 28 days missed) — ~29.8%
- Class 1: High Injury Risk (28+ days missed) — ~70.2%

### Why 28 days as the threshold?

Sports medicine literature uses a **4-week absence** (28 days) as the boundary for a "significant" or "severe" injury — one that materially impacts a team's selection. A shorter threshold (e.g., any injury) would classify ~99%+ of players as injured every season, making the target trivially imbalanced and uninformative.

---

## 2. Dataset

### Source
- **Raw file:** `data/raw/player_injuries.csv`
- **Processed file:** `data/processed/player_injuries_processed.csv`

### Scale

| Property | Value |
|----------|-------|
| Total player-seasons | 1,301 |
| High Injury rate | 70.2% |
| Features (after preprocessing) | 17 |
| Target | `High_Injury` (1 if ≥28 days injured, else 0) |
| Train/Test (chronological) | 964 train / 242 test |
| Train years | 2016–2020 (training partition) |
| Test year | 2020 (held-out season) |

Note: The processed CSV has 1,301 rows; after dropping NaN rows and applying the chronological split, the chronological notebook uses 964/242.

### Raw features (before preprocessing)

The raw CSV contains both **pre-season** attributes and **in-season results** (which would be leakage). The preprocessing stage drops all in-season columns.

**Leakage columns (dropped in preprocessing):**
- `season_days_injured` — This IS the target (or its source); keeping it would be perfect leakage
- `total_days_injured` — Aggregates future injuries
- `season_minutes_played`, `season_games_played`, `season_matches_in_squad` — In-season results
- `total_minutes_played`, `total_games_played` — Can include current season

### Final 17 features (used for training)

| Feature | Description |
|---------|-------------|
| `height_cm` | Player height in centimetres |
| `weight_kg` | Player weight in kilograms |
| `pace` | FIFA pace rating (0–100) |
| `physic` | FIFA physical rating (0–100) |
| `fifa_rating` | Overall FIFA rating (e.g., 78) |
| `age` | Player age at start of season |
| `bmi` | Body Mass Index = weight / (height/100)² |
| `work_rate_numeric` | Encoded work rate: Low=1, Medium=2, High=3 |
| `position_numeric` | Encoded position: GK=1, DF=2, MF=3, FW=4 |
| `cumulative_minutes_played` | Career total minutes played (pre-season) |
| `cumulative_games_played` | Career total games played (pre-season) |
| `cumulative_days_injured` | Career total days injured (pre-season) |
| `minutes_per_game_prev_seasons` | Average minutes per game across prior seasons |
| `avg_days_injured_prev_seasons` | Average days injured per season across prior seasons |
| `avg_games_per_season_prev_seasons` | Average games per season across prior seasons |
| `significant_injury_prev_season` | Binary: 1 if missed 28+ days the previous season |
| `season_days_injured_prev_season` | Number of days injured in the immediately prior season |

---

## 3. Feature Engineering Details

### Target construction

```python
High_Injury = 1 if season_days_injured >= 28 else 0
```

This is computed from the raw `season_days_injured` column, which is then **dropped** before training.

### BMI computation

```python
bmi = weight_kg / (height_cm / 100) ** 2
```

BMI is a derived feature that combines height and weight into a single body composition proxy. While not a perfect measure of athleticism, it captures body size effects on injury vulnerability (e.g., very high BMI may indicate overweight risk; very low BMI may indicate inadequate muscle mass).

### Work rate encoding

The raw `work_rate` column contains text values like "High/Medium" (separated by a slash indicating attacking/defensive work rates). The preprocessing extracts the first component and maps:
- "Low" → 1
- "Medium" → 2
- "High" → 3

### Position encoding

The raw `position` column contains full position names. The preprocessing extracts the first two characters and maps:
- "GK" (Goalkeeper) → 1
- "DF" (Defender) → 2
- "MF" (Midfielder) → 3
- "FW" (Forward/Attacker) → 4

### Missing value imputation

Injury history columns (cumulative stats, prev-season stats) are filled with **0** for players in their first recorded season — interpreted as "no recorded injury history" rather than "missing data". Physical attributes (pace, physic) are imputed where absent.

---

## 4. Notebook Pipeline

### 4.1 `cycle3_exploration_injuries.ipynb` — Data Exploration

**What it does:**
- Loads and inspects the raw injuries CSV
- Examines the schema: which columns are pre-season vs. in-season
- Plots the distribution of `season_days_injured` (highly right-skewed — most players miss 0 days, some miss 200+)
- Tests the sensitivity of the binary threshold: how the High/Low split changes at different day counts
- Identifies the leakage columns that encode in-season information

**Key finding:** Using `season_days_injured > 0` as the threshold would classify 99%+ of players as "injured" (even a single day of minor injury counts). The 28-day threshold produces a more useful 70%/30% split.

### 4.2 `cycle3_preprocessing_injuries.ipynb` — Feature Engineering

**What it does:**
- Drops all leakage columns (listed above)
- Creates `High_Injury` target from `season_days_injured >= 28`
- Applies ordinal encodings for work_rate and position
- Computes BMI
- Imputes missing history columns with 0
- Drops NaN rows (rows with missing values in non-history columns)
- Writes `data/processed/player_injuries_processed.csv`

### 4.3 `cycle3_modelling.ipynb` — Baseline Model Comparison (Random Split)

**What it does:**
- Applies stratified 80/20 random split
- Trains five models: Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM
- Reports AUC-ROC as primary metric

**Results (untuned, random split):**

| Model | Test AUC-ROC |
|-------|-------------|
| Dummy | 0.5000 |
| Logistic Regression | 0.6220 |
| Random Forest | 0.5916 (majority-class collapse) |
| XGBoost | ~0.5900 |
| **LightGBM** | **0.6355** |

**Note on Random Forest:** The untuned RF showed majority-class collapse — it predicted High Injury for nearly all samples. The `balanced` class weighting in tuning fixes this.

### 4.4 `cycle3_tuning.ipynb` — Hyperparameter Tuning (Random Split)

**What it does:**
- `RandomizedSearchCV` (50 combinations × 5-fold stratified CV) for XGBoost, Random Forest, LightGBM
- Tests `scale_pos_weight` at the computed ratio (~0.42) plus fixed alternatives (0.5, 0.637, 1.0)

**Results (tuned, random split):**

| Model | Test AUC-ROC |
|-------|-------------|
| XGBoost Tuned | 0.6558 |
| LightGBM Tuned | 0.6273 (regression from baseline!) |
| Random Forest Tuned | 0.6169 |
| Logistic Regression (baseline) | 0.6220 |

Note: LightGBM overfits on the random split when tuned — `num_leaves=50` on a 1,040-row training set memorises training noise.

### 4.5 `chronological/cycle3_modelling_chronological.ipynb` — Baseline (Chronological)

**What it does:**
- Sorts player-seasons by `start_year`
- Takes first 80% as training (2016–2020), last 20% as test (2020 season)
- Trains and evaluates the same five models

**Why chronological for injury risk?** The same argument as Cycle 1 applies: injury patterns may be temporally structured (e.g., changes in player monitoring, medical practices, or training loads over years). Chronological split ensures the model is tested on future seasons it has never seen.

### 4.6 `chronological/cycle3_tuning_chronological.ipynb` — Tuning (Chronological) ← DEPLOYED MODEL

**What it does:**
- Reconstructs the chronological split from the **raw** CSV (the processed CSV lacks `start_year` needed for temporal splitting)
- Applies the same feature engineering as the preprocessing notebook
- `RandomizedSearchCV` over XGBoost, RF, LightGBM plus Logistic Regression
- Saves the best model as `models/cycle3/cycle3_best_model.pkl`
- Saves scaler as `models/cycle3/cycle3_scaler.pkl`
- Saves feature list as `models/cycle3/cycle3_feature_cols.pkl`

**Final results (tuned, chronological split):**

| Model | Chrono AUC | Random AUC | Delta |
|-------|-----------|------------|-------|
| **LightGBM Tuned** | **0.6800** | 0.6273 | +0.0527 |
| XGBoost Tuned | 0.6723 | 0.6558 | +0.0165 |
| Random Forest Tuned | 0.6668 | 0.6169 | +0.0499 |
| Logistic Regression (baseline) | 0.6263 | 0.6220 | +0.0043 |

**Key observation:** All models perform *better* on the chronological test (held-out 2020 season) than on the random split, which is unusual. This suggests the model's learned patterns transfer well to the 2020 season — injury patterns are temporally stable and learnable from prior seasons.

**LightGBM best hyperparameters (chronological):**
- `subsample`: 0.8
- `scale_pos_weight`: 1.0 (equal weighting chosen over the empirical ~0.42)
- `num_leaves`: 50
- `n_estimators`: 200
- `min_child_samples`: 20
- `max_depth`: 3
- `learning_rate`: 0.01
- `colsample_bytree`: 0.8

### 4.7 `cycle3_explainability.ipynb` — SHAP Analysis

**What it does:**
- Loads the deployed LightGBM model and scaler
- Reconstructs the chronological test set from raw data (same as the tuning notebook)
- Computes SHAP values using `shap.TreeExplainer`
- Saves three plots to `docs/`:

| Plot | File | What it shows |
|------|------|---------------|
| Global importance | `docs/cycle3_shap_global_importance.png` | Which features most affect injury risk predictions |
| Beeswarm summary | `docs/cycle3_shap_summary.png` | Direction and magnitude per feature across all players |
| Waterfall | `docs/cycle3_shap_waterfall.png` | Single-player explanation: how features contributed |

---

## 5. Model Evaluation

### Classification report (LightGBM tuned, chronological)

The deployed model strongly predicts High Injury due to its `scale_pos_weight=1.0` configuration (equal class weighting):

```
              precision    recall  f1-score   support

  Low Injury       0.82      0.19      0.31        93
 High Injury       0.66      0.97      0.79       149

    accuracy                           0.67       242
   macro avg       0.74      0.58      0.55       242
weighted avg       0.72      0.67      0.60       242
```

**Interpreting this:**
- **High Injury recall = 0.97** — The model identifies 97% of players who will have a significant injury. Almost all high-risk players are correctly flagged.
- **Low Injury recall = 0.19** — The model only correctly identifies 19% of genuinely low-risk players. Most low-risk players are incorrectly flagged as high risk.
- **This is a screening tool trade-off:** High recall for the positive class is prioritised over precision. Medical staff would rather over-flag (investigate players who turn out to be fine) than miss at-risk players. AUC (0.68) measures the model's discriminating ability across all thresholds.

### ROC curve

The ROC curve is saved at `docs/cycle3_roc_baseline.png`. It shows the baseline vs. tuned model performance across all classification thresholds.

---

## 6. SHAP Findings

### Most important features (global)

Based on mean absolute SHAP across the chronological test set:

1. **avg_days_injured_prev_seasons** — Past average injury burden is the strongest predictor of future injury. Players who have historically been injury-prone remain so.
2. **cumulative_days_injured** — Career total injury days (similar to above but across full career).
3. **significant_injury_prev_season** — Having missed 28+ days the previous season strongly increases risk.
4. **season_days_injured_prev_season** — Raw injury days from the most recent season.
5. **age** — Older players face higher injury risk (biological wear and recovery time).
6. **cumulative_minutes_played** — Higher career load correlates with increased injury risk (overuse patterns).
7. **BMI** — Extreme values (very high BMI) slightly increase injury risk.
8. **physic (FIFA)** — Physical rating has some predictive value; higher physic correlates with robustness.
9. **position_numeric** — Position has moderate importance; defenders and forwards face different injury profiles.
10. **work_rate_numeric** — Work rate contributes minimally.

### Key SHAP direction patterns

| Feature | High value → | Low value → |
|---------|-------------|------------|
| avg_days_injured_prev_seasons | Strongly increases injury risk | Reduces risk |
| significant_injury_prev_season | Increases risk | Reduces risk |
| age | Increases risk (older players) | — |
| cumulative_minutes_played | Increases risk (accumulated load) | — |
| physic (FIFA) | Slightly reduces risk (more physically robust) | — |

**Key validation:** The SHAP findings align strongly with established sports medicine research:
- Prior injury is the #1 predictor of future injury (not a model artifact — this is the clinical consensus)
- Age-related injury risk increase is well-documented
- Accumulated workload as an overuse injury risk factor is supported by sports science literature

This validates that the model is capturing real, medically meaningful patterns rather than overfitting to noise.

---

## 7. How This Connects to the Full System

### At inference time (API)

When a user calls `POST /predict` with `model="injury"`:

1. The user provides all 17 player features as a JSON object
2. `api/ml/pipeline.py:build_injury_vector()` constructs the feature DataFrame (pass-through — all features are provided directly)
3. The 17-column vector is scaled with the saved `StandardScaler`
4. LightGBM returns `predict_proba()[:, 1]` — the High Injury probability
5. The API returns the probability and a `"High Injury Risk"` / `"Low Injury Risk"` label

### In pipeline mode

When `model="pipeline"`:
- Lists of `home_players` and `away_players` are provided
- The injury model runs on each player
- Results aggregate to: `high_risk_count`, `avg_risk_probability`, `players_assessed`
- The squad with more high-risk players has its match-win probability nudged down (up to ±5%)

### In the Streamlit dashboard (Page 3: Injury Risk)

- The user enters player physical attributes, FIFA ratings, demographics, and injury history
- The injury gauge displays risk probability visually (0–100% scale)
- Optional SHAP bar chart shows which specific factors drove this player's risk score

---

## 8. The Challenge of Injury Prediction

### Why AUC ~0.68 is meaningful, not disappointing

Injury prediction is fundamentally limited by **missing causal factors**:
- **Training load** — How hard a player trained this pre-season is the primary modifiable risk factor, but not measurable from public data
- **Match intensity** — Contact events, high-speed running distance per match
- **Mental health/fatigue** — Psychological factors affect injury risk but are not quantifiable
- **Medical interventions** — Whether a team has top-tier physiotherapy affects injury rates
- **Pitch conditions** — Soft ground, artificial surfaces, late-season fatigue

Published academic literature on injury prediction typically reports AUC 0.60–0.72 using tabular features. This project's AUC of 0.6800 is solidly within this range, achieved with a relatively small dataset (1,301 player-seasons).

### Why the dataset is small

Player injury data requires expensive, privileged sources (club records, medical databases). The ~1,300 player-season dataset used here is the practical limit of open-source football injury data. With 5,000–10,000 player-seasons, model performance would likely improve significantly.

---

## 9. Limitations

1. **Small dataset** — 1,301 player-seasons is genuinely small for machine learning. The model has limited statistical power to detect subtle patterns.

2. **Temporal scope** — Training covers 2016–2020, testing on 2020. Changes in training science, nutrition, or medical protocols post-2020 may shift injury patterns.

3. **No training load data** — The strongest causal predictor of injury is training load intensity, which is not available in any public dataset.

4. **No squad depth data** — Whether a player is the "first choice" starter (and therefore plays through fatigue) or a rotation player significantly affects injury risk.

5. **FIFA attributes as proxy** — `pace`, `physic`, and `fifa_rating` are editorial ratings assigned by game developers, not precise biomechanical measurements. They serve as rough proxies.

6. **High Injury rate (70.2%)** — The majority class is High Injury, which means the model has a strong baseline to predict most players as High Injury. This makes it a screening tool rather than a precise classifier.

7. **Threshold sensitivity** — The 28-day threshold is clinically motivated but arbitrary. Some players may miss 27 days (just under threshold) while being genuinely high-risk; others may miss 35 days due to a single freak accident not predictable from their history.

8. **Deployment gap** — The model was trained on Premier League / top-division data. Its predictions for lower-league players may be less reliable due to different playing styles, medical support, and intensity levels.
