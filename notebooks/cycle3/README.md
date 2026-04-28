# Cycle 3 — Player Injury Risk

Predict whether a player is at high risk of missing **28+ days** in the upcoming season — a binary classification task on physical attributes plus historical injury data.

## Dataset

| Property | Value |
|---|---|
| Source | `data/raw/player_injuries.csv` |
| Processed file | `data/processed/player_injuries_processed.csv` |
| Rows | ~1,300 player-seasons |
| Class balance | ~70% High Injury vs ~30% Low Injury |
| Target | `High_Injury` (1 if `season_days_injured >= 28`) |
| Features | 17 — physical attributes, FIFA ratings, demographics, prior-season summaries, career cumulatives |

## Why a 28-day threshold?

Sports-medicine literature treats a 4-week absence as the boundary for a *significant* injury — long enough to affect a team's season. Earlier work used `season_days_injured > 0`, which classed 99.9% of players as injured and produced a trivially imbalanced target.

## Notebook order

1. **`cycle3_exploration_injuries.ipynb`** — schema, missing-value patterns, distribution of `season_days_injured`, threshold sensitivity.
2. **`cycle3_preprocessing_injuries.ipynb`** — drops post-season leakage columns (`season_days_injured`, `season_minutes_played`, etc.) and identifiers; constructs `High_Injury`; encodes `work_rate` and `position` numerically; computes BMI; imputes missing pace/physic and history columns; writes the processed CSV.
3. **`cycle3_modelling.ipynb`** — random 80/20 stratified split. Trains Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM. Reports AUC-ROC as primary metric.
4. **`cycle3_tuning.ipynb`** — `RandomizedSearchCV` over XGBoost / Random Forest / LightGBM on the random split.
5. **`chronological/cycle3_modelling_chronological.ipynb`** — same models, chronological split by `start_year`.
6. **`chronological/cycle3_tuning_chronological.ipynb`** — chronological tuning. **This is the source of the deployed model.** Considers Logistic Regression, XGBoost, RF, and LightGBM and saves the highest-AUC candidate to `models/cycle3/cycle3_best_model.pkl`.

## Leakage to watch for

The raw CSV contains several columns that describe the season *being predicted*:
`season_days_injured`, `total_days_injured`, `season_minutes_played`, `season_games_played`, `season_matches_in_squad`, `total_minutes_played`, `total_games_played`. These are dropped in preprocessing — only **prior-season** and **career-cumulative** values remain in the feature set.

## Deployed model

| Item | Value |
|---|---|
| Algorithm | LightGBM (tuned) — selected from {LR, XGB, RF, LightGBM} by best chronological AUC = 68% |
| Split | Chronological, by `start_year` |
| Artefact | `models/cycle3/cycle3_best_model.pkl` |
| Scaler | `models/cycle3/cycle3_scaler.pkl` |
| Features | `models/cycle3/cycle3_feature_cols.pkl` |

## Caveats

Injury prediction is a hard problem because the strongest causal factors (training load, contact events, mental fatigue, pitch conditions) are not in any tabular dataset. Treat the output as a screening signal, not a clinical decision.
