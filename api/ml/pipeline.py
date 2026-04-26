from __future__ import annotations

"""
Inference-time feature pipeline wrappers.

Wraps the two feature-engineering steps that must happen at inference time:
  - Cycle 1 (match): merge feature_store stats into the 19-column feature vector
  - Cycle 2 (xG):    compute Distance and Angle from raw X, Y coordinates
  - Cycle 3 (injury): pass-through (all features provided directly)
"""

import math
import numpy as np
import pandas as pd
from api.ml.feature_store import get_team_stats, ROLLING_STAT_KEYS

# Cycle 2 pitch constants (Premier League standard: 105m x 68m)
_PITCH_LENGTH_M = 105.0
_PITCH_WIDTH_M  = 68.0
_POST_LEFT_M    = 30.34  # left post Y in metres
_POST_RIGHT_M   = 37.66  # right post Y in metres
_GOAL_CENTRE_Y  = (_POST_LEFT_M + _POST_RIGHT_M) / 2  # 34.0m


def build_match_vector(
    home_team_id: int,
    away_team_id: int,
    attendance: float,
    feature_cols: list[str],
    home_stats_override: dict | None = None,
    away_stats_override: dict | None = None,
) -> pd.DataFrame:
    """
    Assemble the 19-column feature vector for a match prediction.

    Pulls rolling stats from the feature store unless overrides are supplied.
    Raises ValueError if a team ID is not in the store and no override is given.
    """
    home_stats = home_stats_override if home_stats_override is not None else get_team_stats(home_team_id)
    away_stats = away_stats_override if away_stats_override is not None else get_team_stats(away_team_id)

    if home_stats is None:
        raise ValueError(f"Team ID {home_team_id} not found in feature store.")
    if away_stats is None:
        raise ValueError(f"Team ID {away_team_id} not found in feature store.")

    row = {
        "attendance": attendance,
        "Home Team":  home_team_id,
        "Away Team":  away_team_id,
        **{f"home_{k}": v for k, v in home_stats.items()},
        **{f"away_{k}": v for k, v in away_stats.items()},
    }
    return pd.DataFrame([row])[feature_cols]


def build_xg_vector(
    X: float,
    Y: float,
    left_foot: int,
    right_foot: int,
    header: int,
    first_half: int,
    player_rank: float,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, float, float]:
    """
    Build xG feature vector and compute Distance + Angle from raw coordinates.
    Returns (DataFrame, distance_m, angle_deg).
    """
    x_m = X / 100.0 * _PITCH_LENGTH_M
    y_m = Y / 100.0 * _PITCH_WIDTH_M

    distance_m = float(math.sqrt((x_m - _PITCH_LENGTH_M) ** 2 + (y_m - _GOAL_CENTRE_Y) ** 2))

    dx = _PITCH_LENGTH_M - x_m
    angle_deg = float(abs(math.degrees(
        math.atan2(_POST_RIGHT_M - y_m, dx) - math.atan2(_POST_LEFT_M - y_m, dx)
    )))

    row = {
        "X":           X,
        "Y":           Y,
        "Distance":    distance_m,
        "Angle":       angle_deg,
        "Left_Foot":   left_foot,
        "Right_Foot":  right_foot,
        "Header":      header,
        "First_Half":  first_half,
        "Player_Rank": player_rank,
    }
    return pd.DataFrame([row])[feature_cols], distance_m, angle_deg


def build_injury_vector(data: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Pass-through: all 17 injury features are provided directly."""
    return pd.DataFrame([data])[feature_cols]
