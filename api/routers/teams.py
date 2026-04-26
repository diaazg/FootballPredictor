from fastapi import APIRouter, HTTPException
from api.schemas.team import TeamListResponse, TeamDetailResponse, TeamStats
from api.services.team_service import get_all_teams, get_team

router = APIRouter()


@router.get(
    "/teams",
    response_model=TeamListResponse,
    summary="List all teams with their latest rolling stats",
)
def teams_list():
    """
    Return every team in the feature store along with their most recent
    5-match rolling averages. These are the stats the match prediction
    model will use when you provide a team ID.
    """
    all_teams = get_all_teams()
    return TeamListResponse(
        teams=[TeamStats(**t) for t in all_teams],
        count=len(all_teams),
    )


@router.get(
    "/teams/{team_id}",
    response_model=TeamDetailResponse,
    summary="Get latest rolling stats for one team",
)
def team_stats(team_id: int):
    """
    Return the latest 5-match rolling averages for a specific team.

    `team_id` is the label-encoded integer (1-25) used in `/predict`.
    Returns 404 if the team ID is not in the feature store.
    """
    team = get_team(team_id)
    if team is None:
        raise HTTPException(
            status_code=404,
            detail=f"Team ID {team_id} not found. Use GET /teams to see available IDs.",
        )
    return TeamDetailResponse(**team)
