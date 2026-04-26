"""
Prediction service: business logic for all three cycles.

Flow for each prediction type:
  1. Pull ModelEntry from registry (pre-loaded, no disk I/O)
  2. Build feature vector via pipeline
  3. Scale → predict
  4. Return structured result dict
"""

import numpy as np
from api.services.model_registry import get_entry
from api.ml.pipeline import build_match_vector, build_xg_vector, build_injury_vector

# Cycle 1 outcome labels (model outputs 0/1/2)
_MATCH_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def _confidence(max_prob: float) -> str:
    if max_prob >= 0.60:
        return "high"
    if max_prob >= 0.45:
        return "medium"
    return "low"


def predict_match(
    home_team_id: int,
    away_team_id: int,
    attendance: float,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
) -> dict:
    entry = get_entry("match")
    X = build_match_vector(home_team_id, away_team_id, attendance, entry.feature_cols,
                           home_stats_override=home_stats, away_stats_override=away_stats)
    X_scaled = entry.scaler.transform(X)
    code = int(entry.model.predict(X_scaled)[0])
    probs = entry.model.predict_proba(X_scaled)[0]

    prob_home = round(float(probs[2]), 4)
    prob_draw = round(float(probs[1]), 4)
    prob_away = round(float(probs[0]), 4)
    max_prob  = max(prob_home, prob_draw, prob_away)

    return {
        "prediction":    _MATCH_LABELS[code],
        "probabilities": {"home_win": prob_home, "draw": prob_draw, "away_win": prob_away},
        "confidence":    _confidence(max_prob),
        "model_used":    "xgboost_tuned",
        "cycle":         1,
    }


def predict_xg(
    X: float,
    Y: float,
    left_foot: int,
    right_foot: int,
    header: int,
    first_half: int,
    player_rank: float,
) -> dict:
    entry = get_entry("xg")
    feat_df, distance_m, angle_deg = build_xg_vector(
        X, Y, left_foot, right_foot, header, first_half, player_rank,
        entry.feature_cols,
    )
    X_scaled = entry.scaler.transform(feat_df)
    xg_prob = round(float(entry.model.predict_proba(X_scaled)[0][1]), 4)
    goal = bool(entry.model.predict(X_scaled)[0] == 1)

    return {
        "prediction":    "Goal" if goal else "No Goal",
        "probabilities": {"goal": xg_prob, "no_goal": round(1 - xg_prob, 4)},
        "confidence":    _confidence(max(xg_prob, 1 - xg_prob)),
        "xg":            xg_prob,
        "distance_m":    round(distance_m, 2),
        "angle_deg":     round(angle_deg, 2),
        "model_used":    "xgboost_tuned",
        "cycle":         2,
    }


def predict_injury(features: dict) -> dict:
    entry = get_entry("injury")
    X = build_injury_vector(features, entry.feature_cols)
    X_scaled = entry.scaler.transform(X)
    probs = entry.model.predict_proba(X_scaled)[0]

    prob_high = round(float(probs[1]), 4)
    prob_low  = round(float(probs[0]), 4)
    label = "High Injury Risk" if prob_high >= 0.5 else "Low Injury Risk"

    return {
        "prediction":    label,
        "probabilities": {"high_injury": prob_high, "low_injury": prob_low},
        "confidence":    _confidence(max(prob_high, prob_low)),
        "model_used":    "logistic_regression",
        "cycle":         3,
    }
