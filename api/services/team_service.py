from __future__ import annotations

"""
Team service: exposes rolling feature data from the feature store.
"""

from api.ml.feature_store import get_team_stats, list_teams, ROLLING_STAT_KEYS, TEAM_NAMES


def get_all_teams() -> list[dict]:
    return [
        {"team_id": tid, "name": TEAM_NAMES.get(tid, f"Team {tid}"), "stats": get_team_stats(tid)}
        for tid in list_teams()
    ]


def get_team(team_id: int) -> dict | None:
    stats = get_team_stats(team_id)
    if stats is None:
        return None
    return {
        "team_id":  team_id,
        "name":     TEAM_NAMES.get(team_id, f"Team {team_id}"),
        "stats":    stats,
        "stat_keys": ROLLING_STAT_KEYS,
    }
