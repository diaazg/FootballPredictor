from pydantic import BaseModel, Field


class TeamStats(BaseModel):
    team_id: int  = Field(..., description="Label-encoded team ID (1-25)")
    name:    str  = Field("", description="Human-readable team name")
    stats:   dict = Field(..., description="Latest 5-match rolling averages for this team")


class TeamDetailResponse(TeamStats):
    stat_keys: list[str] = Field(..., description="Ordered list of stat names")


class TeamListResponse(BaseModel):
    teams: list[TeamStats]
    count: int
