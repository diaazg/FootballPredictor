from __future__ import annotations
from pydantic import BaseModel, Field


class TeamStats(BaseModel):
    team_id: int  = Field(..., description="Label-encoded team ID (alphabetic over PL teams 2000-2018)")
    name:    str  = Field("", description="Human-readable team name")
    stats:   dict = Field(..., description="Latest snapshot of the team's season-to-date stats")


class TeamDetailResponse(TeamStats):
    stat_keys: list[str] = Field(..., description="Ordered list of stat names")


class TeamListResponse(BaseModel):
    teams: list[TeamStats]
    count: int
