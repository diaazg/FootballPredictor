# Cycle 2 — Expected Goals (xG)

Estimate the probability that a single shot results in a goal — a binary classification task on Wyscout event data.

## Dataset

| Property | Value |
|---|---|
| Source | `data/raw/events_England.json` (Wyscout open-data, England 2017/18) plus `playerank.json` |
| Processed file | `data/processed/wyscout_shots_processed.csv` |
| Rows after filtering | 8,451 open-play shots |
| Class balance | ~89% No Goal vs ~11% Goal |
| Target | `Goal` (1 = goal, 0 = no goal) |

Pre-shot features only — no post-shot tags (e.g. where in the goal the ball ended) so there is no leakage.

| Feature | Source |
|---|---|
| `X`, `Y` | Shot location (Wyscout 0–100 pitch coordinates) |
| `Distance` | Computed from X, Y to goal centre |
| `Angle` | Computed from X, Y to both posts |
| `Left_Foot`, `Right_Foot`, `Header` | Wyscout body-part tags (401/402/403) |
| `First_Half` | Match period |
| `Player_Rank` | `playerankScore` joined on `playerId` (proxy for player quality) |

## Notebook order

1. **`cycle2_exploration_wyscout.ipynb`** — inspect raw events, identify leakage tags, decide on open-play-shots scope.
2. **`cycle2_preprocessing_wyscout.ipynb`** — filter to shots, derive Distance/Angle/foot/period features, merge `playerankScore`, write the processed CSV.
3. **`cycle2_modelling.ipynb`** — random 80/20 stratified split. Compares Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM. Reports both Accuracy and AUC-ROC; AUC is primary because of the 9:1 imbalance (the dummy gets ~89% accuracy but AUC = 0.5).
4. **`cycle2_tuning.ipynb`** — `RandomizedSearchCV` (50 iter × 5 folds) over XGBoost / Random Forest / LightGBM with `scale_pos_weight` swept to address imbalance. Saves the best model to `models/cycle2/cycle2_best_model.pkl`.

## Why AUC-ROC, not accuracy

A trivial "always predict No Goal" classifier achieves ~89% accuracy. AUC-ROC measures how well the model **ranks** goals above non-goals, independent of any threshold — that is the property a deployed xG model needs.

## Deployed model

| Item | Value |
|---|---|
| Algorithm | XGBoost (tuned) |
| Split | Random 80/20, stratified on `Goal` |
| Test AUC-ROC | **0.8183** (vs 0.5 dummy baseline) |
| Artefact | `models/cycle2/cycle2_best_model.pkl` |
| Scaler | `models/cycle2/cycle2_scaler.pkl` (`StandardScaler` fit on training features) |
| Features | `models/cycle2/cycle2_feature_cols.pkl` |

## Caveats / scope

- Open-play shots only. Free kicks, penalties, and set-piece headers behave differently and would warrant their own model.
- England (Premier League) only. Adding France / Germany / Italy / Spain would 5–6× the training set.
- Player skill is captured only through `playerankScore` — no team-level context.
