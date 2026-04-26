"""
Pydantic I/O contracts for POST /predict.

Uses a discriminated union on the `model` field so a single endpoint
handles all three prediction cycles. FastAPI generates separate schema
definitions in the OpenAPI docs for each variant.
"""

from __future__ import annotations
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field


# ── Input schemas ─────────────────────────────────────────────────────────────

class MatchPredictionRequest(BaseModel):
    model: Literal["match"]
    home_team_id: int   = Field(..., ge=1, le=25, example=12,
                                description="Label-encoded home team ID (1-25)")
    away_team_id: int   = Field(..., ge=1, le=25, example=7,
                                description="Label-encoded away team ID (1-25)")
    attendance:   float = Field(0, ge=0, example=45000,
                                description="Match attendance (0 = use dataset mean)")
    home_stats:   Optional[dict] = Field(None,
                                description="Override home team rolling stats (8 avg_*_5 keys). "
                                            "If omitted, stats are loaded from the feature store.")
    away_stats:   Optional[dict] = Field(None,
                                description="Override away team rolling stats (8 avg_*_5 keys). "
                                            "If omitted, stats are loaded from the feature store.")


class XGPredictionRequest(BaseModel):
    model:       Literal["xg"]
    X:           float = Field(..., ge=0, le=100, example=85.0,
                               description="Shot X coordinate (0-100 pct of pitch length; 100 = goal line)")
    Y:           float = Field(..., ge=0, le=100, example=50.0,
                               description="Shot Y coordinate (0-100 pct of pitch width; 50 = centre)")
    left_foot:   int   = Field(..., ge=0, le=1,   example=0,
                               description="1 if taken with left foot (tag 401)")
    right_foot:  int   = Field(..., ge=0, le=1,   example=1,
                               description="1 if taken with right foot (tag 402)")
    header:      int   = Field(..., ge=0, le=1,   example=0,
                               description="1 if header or body shot (tag 403)")
    first_half:  int   = Field(..., ge=0, le=1,   example=1,
                               description="1 if first half, 0 if second half")
    player_rank: float = Field(...,               example=6.5,
                               description="playerankScore — proxy for player quality")


class InjuryPredictionRequest(BaseModel):
    model: Literal["injury"]
    height_cm:                         float = Field(..., example=181.0)
    weight_kg:                         float = Field(..., example=76.0)
    pace:                              float = Field(..., example=72.0)
    physic:                            float = Field(..., example=75.0)
    fifa_rating:                       float = Field(..., example=78.0)
    age:                               float = Field(..., example=26.0)
    bmi:                               float = Field(..., example=23.2)
    work_rate_numeric:                 float = Field(..., example=2.0,
                                                     description="Low=1, Medium=2, High=3")
    position_numeric:                  float = Field(..., example=3.0)
    cumulative_minutes_played:         float = Field(..., example=8200.0)
    cumulative_games_played:           float = Field(..., example=95.0)
    cumulative_days_injured:           float = Field(..., example=45.0)
    minutes_per_game_prev_seasons:     float = Field(..., example=72.0)
    avg_days_injured_prev_seasons:     float = Field(..., example=8.5)
    avg_games_per_season_prev_seasons: float = Field(..., example=28.0)
    significant_injury_prev_season:    int   = Field(..., ge=0, le=1, example=0,
                                                     description="1 if 28+ day injury last season")
    season_days_injured_prev_season:   float = Field(..., example=12.0)


PredictionRequest = Annotated[
    Union[MatchPredictionRequest, XGPredictionRequest, InjuryPredictionRequest],
    Field(discriminator="model"),
]


# ── Output schemas ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction:    str         = Field(..., description="Human-readable outcome label")
    probabilities: dict        = Field(..., description="Probability per class")
    confidence:    str         = Field(..., description="high / medium / low")
    model_used:    str         = Field(..., description="Identifier of the model that ran")
    cycle:         int         = Field(..., description="Which prediction cycle (1, 2, or 3)")
    xg:            Optional[float] = Field(None, description="xG probability (Cycle 2 only)")
    distance_m:    Optional[float] = Field(None, description="Shot distance in metres (Cycle 2 only)")
    angle_deg:     Optional[float] = Field(None, description="Shot angle in degrees (Cycle 2 only)")
