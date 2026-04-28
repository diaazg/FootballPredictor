from __future__ import annotations

"""
Per-team latest-stats cache for Cycle 1 (match outcome).

Loaded once at startup from premier_league_matches_processed.csv.
For each team, captures the snapshot of its season-to-date stats from its
most recent appearance in the dataset:

  - goals_scored, goals_conceded, points, form_pts, gd
  - win_streak_3, win_streak_5, loss_streak_3, loss_streak_5
  - m1..m5 (last 5 results, encoded 0/1/3 = L/D/W)

Also exposes team-id → team-name lookup, derived from the raw CSV using the
same alphabetic encoding as the preprocessing notebook.
"""

import os
import pandas as pd

_PROCESSED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed",
    "premier_league_matches_processed.csv",
)
_RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw",
    "premier_league_matches.csv",
)

# Snapshot fields captured per team
TEAM_STAT_KEYS = [
    "goals_scored", "goals_conceded", "points", "form_pts", "gd",
    "win_streak_3", "win_streak_5", "loss_streak_3", "loss_streak_5",
    "m1", "m2", "m3", "m4", "m5",
]

_store: dict[int, dict[str, float]] = {}
_team_names: dict[int, str] = {}
_latest_mw: int = 1
_ready: bool = False


def _build_team_id_map() -> dict[str, int]:
    """Reproduce the preprocessing notebook's alphabetic team encoding."""
    raw = pd.read_csv(_RAW_PATH)
    all_teams = sorted(pd.concat([raw["HomeTeam"], raw["AwayTeam"]]).unique())
    return {team: idx for idx, team in enumerate(all_teams)}


def build_feature_store() -> None:
    global _ready, _latest_mw

    # Team ID → name lookup
    team_map = _build_team_id_map()
    _team_names.update({idx: name for name, idx in team_map.items()})

    df = pd.read_csv(_PROCESSED_PATH)

    # Build a per-team row for every match appearance (home and away combined)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "team":           int(r["HomeTeam"]),
            "season":         int(r["Season"]),
            "mw":             int(r["MW"]),
            "goals_scored":   float(r["HTGS"]),
            "goals_conceded": float(r["HTGC"]),
            "points":         float(r["HTP"]),
            "form_pts":       float(r["HTFormPts"]),
            "gd":             float(r["HTGD"]),
            "win_streak_3":   int(r["HTWinStreak3"]),
            "win_streak_5":   int(r["HTWinStreak5"]),
            "loss_streak_3":  int(r["HTLossStreak3"]),
            "loss_streak_5":  int(r["HTLossStreak5"]),
            "m1": int(r["HM1"]), "m2": int(r["HM2"]), "m3": int(r["HM3"]),
            "m4": int(r["HM4"]), "m5": int(r["HM5"]),
        })
        rows.append({
            "team":           int(r["AwayTeam"]),
            "season":         int(r["Season"]),
            "mw":             int(r["MW"]),
            "goals_scored":   float(r["ATGS"]),
            "goals_conceded": float(r["ATGC"]),
            "points":         float(r["ATP"]),
            "form_pts":       float(r["ATFormPts"]),
            "gd":             float(r["ATGD"]),
            "win_streak_3":   int(r["ATWinStreak3"]),
            "win_streak_5":   int(r["ATWinStreak5"]),
            "loss_streak_3":  int(r["ATLossStreak3"]),
            "loss_streak_5":  int(r["ATLossStreak5"]),
            "m1": int(r["AM1"]), "m2": int(r["AM2"]), "m3": int(r["AM3"]),
            "m4": int(r["AM4"]), "m5": int(r["AM5"]),
        })

    tm = pd.DataFrame(rows)
    # Latest per team = max (season, mw) across appearances
    tm["rank_key"] = tm["season"] * 100 + tm["mw"]
    latest = tm.sort_values("rank_key").groupby("team").last().reset_index()

    for _, row in latest.iterrows():
        team_id = int(row["team"])
        _store[team_id] = {k: float(row[k]) for k in TEAM_STAT_KEYS}

    _latest_mw = int(latest["mw"].max())
    _ready = True


def get_team_stats(team_id: int) -> dict[str, float] | None:
    return _store.get(team_id)


def get_team_name(team_id: int) -> str | None:
    return _team_names.get(team_id)


def list_teams() -> list[int]:
    return sorted(_store.keys())


def latest_mw() -> int:
    return _latest_mw


def is_ready() -> bool:
    return _ready
