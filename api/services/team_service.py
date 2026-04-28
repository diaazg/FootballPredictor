from __future__ import annotations

"""
Team service: exposes per-team latest snapshot stats from the feature store.
"""

from api.ml.feature_store import (
    get_team_stats,
    get_team_name,
    list_teams,
    TEAM_STAT_KEYS,
)


def get_all_teams() -> list[dict]:
    return [
        {
            "team_id": tid,
            "name":    get_team_name(tid) or f"Team {tid}",
            "stats":   get_team_stats(tid),
        }
        for tid in list_teams()
    ]


def get_team(team_id: int) -> dict | None:
    stats = get_team_stats(team_id)
    if stats is None:
        return None
    return {
        "team_id":   team_id,
        "name":      get_team_name(team_id) or f"Team {team_id}",
        "stats":     stats,
        "stat_keys": TEAM_STAT_KEYS,
    }
