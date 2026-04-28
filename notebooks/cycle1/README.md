# Cycle 1 — Match Outcome Prediction

Predict the result of a Premier League match: **Home Win / Draw / Away Win**.

## Dataset

| Property | Value |
|---|---|
| Source | `data/raw/premier_league_matches.csv` |
| Processed file | `data/processed/premier_league_matches_processed.csv` |
| Coverage | Premier League seasons 2000–2018 |
| Rows | 6,840 matches |
| Features | 33 (form, points, streaks, last-5 results, goal difference, team identity) |
| Target | `FTR` (0 = Away Win, 1 = Draw, 2 = Home Win) |

## Notebook order

Run the notebooks in this directory top-to-bottom:

1. **`cycle1_exploration.ipynb`** — initial inspection of the raw CSV: target encoding, leakage candidates, missing values, scale issues.
2. **`cycle1_preprocessing.ipynb`** — applies the fixes identified in exploration:
   - reconstructs the 3-class `FTR` from `FTHG`/`FTAG`, then drops the goal columns (leakage),
   - encodes form columns (`HM1–HM5`, `AM1–AM5`) as W=3 / D=1 / L=0 / M=0,
   - label-encodes `HomeTeam`/`AwayTeam`,
   - extracts `Season` from `Date`, drops `Date`,
   - writes `data/processed/premier_league_matches_processed.csv`.
3. **`cycle1_modelling.ipynb`** — random 80/20 split. Trains and compares: Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM. Establishes the dummy floor (~46% — Home Win is the majority class).
4. **`cycle1_tuning.ipynb`** — `RandomizedSearchCV` over XGBoost / Random Forest / LightGBM on the random split.
5. **`chronological/cycle1_modelling_chronological.ipynb`** — same models, chronological 80/20 split (last 20% of seasons held out).
6. **`chronological/cycle1_tuning_chronological.ipynb`** — chronological tuning. **This is the source of the deployed model.** Saves `models/cycle1/cycle1_xgb_best.pkl` and `cycle1_feature_cols.pkl`.
7. **`cycle1_explainability.ipynb`** — SHAP analysis on the deployed model. Loads from disk and runs `TreeExplainer` to produce global importance, per-class beeswarm summaries, and a single-prediction waterfall plot. Outputs PNGs to `docs/`.

## Why two splits?

The random split is a familiar baseline but leaks seasons across train and test. For deployment we use the **chronological** split — train on earlier seasons, test on the held-out final 20% — because that mirrors how the model is used: predicting the next match using past data.

## Deployed model

| Item | Value |
|---|---|
| Algorithm | XGBoost (tuned) |
| Split | Chronological |
| Test accuracy | **52.85%** (vs ~46% dummy baseline) |
| Artefact | `models/cycle1/cycle1_xgb_best.pkl` |
| Features | `models/cycle1/cycle1_feature_cols.pkl` |
| Scaler | None — XGBoost was trained on raw features, no scaling required at inference time |

## SHAP outputs

Saved by `cycle1_explainability.ipynb` to `docs/`:
- `cycle1_shap_global_importance.png` — mean |SHAP| across all classes
- `cycle1_shap_summary_home_win.png`, `..._draw.png`, `..._away_win.png` — beeswarm per class
- `cycle1_shap_waterfall.png` — single-prediction decomposition
