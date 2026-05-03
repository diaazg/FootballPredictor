from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.schemas.team import TeamListResponse, TeamDetailResponse, TeamStats
from api.services.team_service import get_all_teams, get_team

router = APIRouter()


@router.get(
    "/teams",
    response_model=TeamListResponse,
    summary="List all teams with their latest snapshot stats",
)
def teams_list():
    """
    Return every team in the feature store along with their most recent
    season-to-date stats (goals, points, form, streaks, last 5 results).
    These are the stats the match prediction model will use when you provide a team ID.
    """
    all_teams = get_all_teams()
    return TeamListResponse(
        teams=[TeamStats(**t) for t in all_teams],
        count=len(all_teams),
    )


@router.get(
    "/teams/{team_id}",
    response_model=TeamDetailResponse,
    summary="Get latest snapshot stats for one team",
)
def team_stats(team_id: int):
    """
    Return the latest snapshot of a team's season-to-date stats.

    `team_id` is the label-encoded integer used in `/predict`.
    Returns 404 if the team ID is not in the feature store.
    """
    team = get_team(team_id)
    if team is None:
        raise HTTPException(
            status_code=404,
            detail=f"Team ID {team_id} not found. Use GET /teams to see available IDs.",
        )
    return TeamDetailResponse(**team)
