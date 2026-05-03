from __future__ import annotations
"""
Prediction service: business logic for all three cycles.

Flow for each prediction type:
  1. Pull ModelEntry from registry (pre-loaded, no disk I/O)
  2. Build feature vector via pipeline
  3. Scale → predict
  4. Return structured result dict

Pipeline mode (predict_pipeline) chains all three:
  shots  → xG model × N  → aggregate team xG stats
  players → injury model × N → aggregate squad risk stats
  team IDs → match model  → base probabilities
  aggregates → _adjust_probs → final probabilities
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
    mw: int | None = None,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
) -> dict:
    entry = get_entry("match")
    X = build_match_vector(home_team_id, away_team_id, entry.feature_cols,
                           mw=mw,
                           home_stats_override=home_stats,
                           away_stats_override=away_stats)
    code = int(entry.model.predict(X)[0])
    probs = entry.model.predict_proba(X)[0]

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


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _aggregate_xg(shots: list) -> dict | None:
    """
    Run the xG model on every shot in `shots` and return aggregated team stats.

    `shots` is a list of ShotInput Pydantic objects (or plain dicts with the
    same keys). Returns None if the list is empty.

    Returned dict keys:
        total_xg       — sum of all shot xG values; proxy for team attacking threat
        avg_xg_per_shot — quality of chance creation (higher = better positions)
        shot_count     — number of shots assessed
        shot_xg_values — individual xG for each shot (for transparency)
    """
    if not shots:
        return None
    xg_values = []
    for shot in shots:
        s = shot.model_dump() if hasattr(shot, "model_dump") else shot
        result = predict_xg(
            s["X"], s["Y"],
            s.get("left_foot", 0), s.get("right_foot", 1),
            s.get("header", 0),    s.get("first_half", 1),
            s.get("player_rank", 7.0),
        )
        xg_values.append(result["xg"])
    return {
        "total_xg":        round(sum(xg_values), 4),
        "avg_xg_per_shot": round(sum(xg_values) / len(xg_values), 4),
        "shot_count":      len(xg_values),
        "shot_xg_values":  [round(v, 4) for v in xg_values],
    }


def _aggregate_injury(players: list) -> dict | None:
    """
    Run the injury model on every player in `players` and return squad risk stats.

    `players` is a list of PlayerInput Pydantic objects (or plain dicts).
    Returns None if the list is empty.

    Returned dict keys:
        high_risk_count      — players where injury probability ≥ 0.5
        avg_risk_probability — mean injury probability across all players
        players_assessed     — total players evaluated
        player_risk_values   — individual probability for each player (for transparency)
    """
    if not players:
        return None
    risk_values = []
    for player in players:
        p = player.model_dump() if hasattr(player, "model_dump") else player
        result = predict_injury(p)
        risk_values.append(result["probabilities"]["high_injury"])
    high_risk = sum(1 for v in risk_values if v >= 0.5)
    return {
        "high_risk_count":      high_risk,
        "avg_risk_probability": round(sum(risk_values) / len(risk_values), 4),
        "players_assessed":     len(risk_values),
        "player_risk_values":   [round(v, 4) for v in risk_values],
    }


def _adjust_probs(
    base: dict,
    home_xg: dict | None,
    away_xg: dict | None,
    home_inj: dict | None,
    away_inj: dict | None,
) -> tuple[dict, list[str]]:
    """
    Nudge match probabilities based on xG quality and injury risk.

    Adjustments are intentionally small (capped at ±0.05 each) so the
    match model's learned probabilities always dominate. After adjustment
    the three probabilities are re-normalised to sum to 1.0.

    Returns (adjusted_probs_dict, list_of_explanation_strings).
    """
    p_home = base["home_win"]
    p_draw = base["draw"]
    p_away = base["away_win"]
    factors: list[str] = []

    # ── xG adjustment ─────────────────────────────────────────────────────────
    # Logic: if home team created higher-quality chances (higher avg xG/shot),
    # nudge home win probability up and away win probability down.
    if home_xg and away_xg:
        home_avg = home_xg["avg_xg_per_shot"]
        away_avg = away_xg["avg_xg_per_shot"]
        diff = home_avg - away_avg          # positive = home creates better chances
        adj  = max(-0.05, min(0.05, diff * 0.5))   # scale and cap at ±5%
        p_home += adj
        p_away -= adj
        direction = "home" if adj > 0 else "away" if adj < 0 else "neither"
        factors.append(
            f"xG quality favours {direction} "
            f"(home {home_avg:.3f} vs away {away_avg:.3f} avg xG/shot, "
            f"home total {home_xg['total_xg']:.2f} vs away {away_xg['total_xg']:.2f}; "
            f"prob adjustment {adj:+.3f})"
        )

    # ── Injury adjustment ──────────────────────────────────────────────────────
    # Logic: more high-risk players in a squad = disadvantage for that team.
    # Each extra high-risk player on the home side reduces home win prob by 0.02.
    if home_inj and away_inj:
        home_risk = home_inj["high_risk_count"]
        away_risk = away_inj["high_risk_count"]
        net = away_risk - home_risk          # positive = home has fewer at-risk players
        adj = max(-0.05, min(0.05, net * 0.02))
        p_home += adj
        p_away -= adj
        if abs(adj) > 0.001:
            direction = "home" if adj > 0 else "away"
            factors.append(
                f"Injury load favours {direction} "
                f"(home {home_risk} vs away {away_risk} high-risk players; "
                f"prob adjustment {adj:+.3f})"
            )
        else:
            factors.append(
                f"Similar injury load "
                f"(home {home_risk} vs away {away_risk} high-risk players; no adjustment)"
            )

    if not factors:
        factors.append(
            "No xG or injury data supplied — returning base match model probabilities unchanged"
        )

    # Re-normalise so probabilities still sum to 1
    total = p_home + p_draw + p_away
    adjusted = {
        "home_win": round(max(0.0, p_home / total), 4),
        "draw":     round(max(0.0, p_draw  / total), 4),
        "away_win": round(max(0.0, p_away  / total), 4),
    }
    return adjusted, factors


def predict_pipeline(
    home_team_id:  int,
    away_team_id:  int,
    mw:            int | None  = None,
    home_stats:    dict | None = None,
    away_stats:    dict | None = None,
    home_shots:    list | None = None,
    away_shots:    list | None = None,
    home_players:  list | None = None,
    away_players:  list | None = None,
) -> dict:
    """
    Integrated pipeline: xG model + injury model + match model in one call.

    Step 1 — Base match prediction (Cycle 1 model, unchanged).
    Step 2 — xG aggregation: run Cycle 2 model on every shot in each list.
    Step 3 — Injury aggregation: run Cycle 3 model on every player in each list.
    Step 4 — Adjustment: nudge base probabilities by xG and injury factors.

    All three steps use the same already-loaded model artefacts; no extra disk I/O.
    """
    # Step 1: base match prediction (identical to model="match")
    match  = predict_match(home_team_id, away_team_id, mw, home_stats, away_stats)
    base_probs = match["probabilities"]

    # Step 2: xG aggregation (skipped if no shots supplied)
    home_xg = _aggregate_xg(home_shots) if home_shots else None
    away_xg = _aggregate_xg(away_shots) if away_shots else None

    # Step 3: injury aggregation (skipped if no players supplied)
    home_inj = _aggregate_injury(home_players) if home_players else None
    away_inj = _aggregate_injury(away_players) if away_players else None

    # Step 4: adjust + explain
    adjusted_probs, factors = _adjust_probs(base_probs, home_xg, away_xg, home_inj, away_inj)

    # Determine final outcome label from adjusted probabilities
    label_map = {"home_win": "Home Win", "draw": "Draw", "away_win": "Away Win"}
    best_key  = max(adjusted_probs, key=adjusted_probs.get)
    max_prob  = adjusted_probs[best_key]

    return {
        "prediction":         label_map[best_key],
        "probabilities":      adjusted_probs,
        "base_probabilities": base_probs,
        "confidence":         _confidence(max_prob),
        "model_used":         "pipeline_v1 (match + xG + injury)",
        "cycle":              1,
        "xg_analysis": (
            {"home": home_xg, "away": away_xg}
            if (home_xg or away_xg) else None
        ),
        "injury_analysis": (
            {"home": home_inj, "away": away_inj}
            if (home_inj or away_inj) else None
        ),
        "pipeline_factors": factors,
    }


# ─────────────────────────────────────────────────────────────────────────────

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
