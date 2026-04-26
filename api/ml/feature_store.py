from __future__ import annotations

"""
Rolling feature cache per team.

Loaded once at startup from skysports_match_stats_cleaned.csv.
Replicates the rolling window logic from cycle1_feature_engineering_skysports.ipynb:
  - Build a unified per-team match history (home + away combined)
  - Compute shift(1).rolling(5, min_periods=1).mean() per team
  - Store each team's latest rolling stats

Keys: int team ID (1-25)
Values: dict of {stat_name -> float}
"""

import os
import pandas as pd

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed",
    "skysports_match_stats_cleaned.csv"
)

# Column mapping: raw stat column → generic stat name
_HOME_COLS = [
    "home_possessions", "home_shots", "home_on", "home_pass",
    "home_tackles", "home_corners", "home_fouls", "home_yellow",
]
_AWAY_COLS = [
    "away_possessions", "away_shots", "away_on", "away_pass",
    "away_tackles", "away_corners", "away_fouls", "away_yellow",
]
_STAT_NAMES = [
    "possession", "shots", "shots_on_target", "pass_accuracy",
    "tackles", "corners", "fouls", "yellow_cards",
]

# The final rolling feature names (as expected by the Cycle 1 model)
ROLLING_STAT_KEYS = [f"avg_{s}_5" for s in _STAT_NAMES]

TEAM_NAMES: dict[int, str] = {
    1: "Manchester City",
    2: "Arsenal",
    3: "Manchester United",
    4: "Newcastle United",
    5: "Liverpool",
    6: "Brighton And Hove Albion",
    7: "Aston Villa",
    8: "Tottenham Hotspur",
    9: "Brentford",
    10: "Fulham",
    11: "Crystal Palace",
    12: "Chelsea",
    13: "Wolverhampton Wanderers",
    14: "West Ham United",
    15: "Bournemouth",
    16: "Nottingham Forest",
    17: "Everton",
    18: "Leicester City",
    19: "Leeds United",
    20: "Southampton",
    21: "Watford",
    22: "Norwich City",
    23: "Burnley",
    24: "West Bromwich Albion",
    25: "Sheffield United",
}

_store: dict[int, dict[str, float]] = {}
_ready: bool = False


def build_feature_store() -> None:
    global _ready

    df = pd.read_csv(_DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["match_idx"] = df.index

    # Build team match history — one row per team per match
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "match_idx": row["match_idx"],
            "date":      row["date"],
            "team":      row["Home Team"],
            "side":      "home",
            **{stat: row[home_col] for stat, home_col in zip(_STAT_NAMES, _HOME_COLS)},
        })
        rows.append({
            "match_idx": row["match_idx"],
            "date":      row["date"],
            "team":      row["Away Team"],
            "side":      "away",
            **{stat: row[away_col] for stat, away_col in zip(_STAT_NAMES, _AWAY_COLS)},
        })

    tm = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)

    # Rolling averages: exclude current match result (shift 1 before rolling)
    for stat in _STAT_NAMES:
        tm[f"avg_{stat}_5"] = (
            tm.groupby("team")[stat]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

    # For each team take their most recent entry (latest rolling snapshot)
    latest = tm.sort_values("date").groupby("team").last().reset_index()

    for _, row in latest.iterrows():
        team_id = int(row["team"])
        _store[team_id] = {
            key: round(float(row[key]), 4) if pd.notna(row[key]) else 0.0
            for key in ROLLING_STAT_KEYS
        }

    _ready = True


def get_team_stats(team_id: int) -> dict[str, float] | None:
    return _store.get(team_id)


def list_teams() -> list[int]:
    return sorted(_store.keys())


def is_ready() -> bool:
    return _ready
