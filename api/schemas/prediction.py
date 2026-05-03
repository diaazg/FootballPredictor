"""
Pydantic I/O contracts for POST /predict.

Uses a discriminated union on the `model` field so a single endpoint
handles all three prediction cycles. FastAPI generates separate schema
definitions in the OpenAPI docs for each variant.

Pipeline mode (model="pipeline") runs all three models in sequence:
  shot list → xG model → team xG aggregates
  player list → injury model → squad risk aggregates
  team IDs → match model → base probabilities
  aggregates → adjustment layer → final probabilities
"""

from __future__ import annotations
from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ── Input schemas ─────────────────────────────────────────────────────────────

class MatchPredictionRequest(BaseModel):
    model: Literal["match"]
    home_team_id: int   = Field(..., ge=0, le=43, example=24,
                                description="Label-encoded home team ID (0-43; alphabetic over all PL teams 2000-2018)")
    away_team_id: int   = Field(..., ge=0, le=43, example=23,
                                description="Label-encoded away team ID (0-43)")
    mw:           Optional[int] = Field(None, ge=1, le=38, example=20,
                                description="Matchweek (1-38). Defaults to latest matchweek in the dataset.")
    home_stats:   Optional[dict] = Field(None,
                                description="Override home team snapshot stats. Keys: goals_scored, "
                                            "goals_conceded, points, form_pts, gd, win_streak_3, "
                                            "win_streak_5, loss_streak_3, loss_streak_5, m1..m5.")
    away_stats:   Optional[dict] = Field(None,
                                description="Override away team snapshot stats (same keys as home_stats).")


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


# ── Pipeline helper input types ───────────────────────────────────────────────

class ShotInput(BaseModel):
    """One shot event used to compute xG inside the pipeline."""
    X:           float = Field(...,  ge=0, le=100, example=82.0,
                               description="Shot X coordinate (0-100, 100 = goal line)")
    Y:           float = Field(...,  ge=0, le=100, example=51.0,
                               description="Shot Y coordinate (0-100, 50 = centre)")
    left_foot:   int   = Field(0,   ge=0, le=1)
    right_foot:  int   = Field(1,   ge=0, le=1)
    header:      int   = Field(0,   ge=0, le=1)
    first_half:  int   = Field(1,   ge=0, le=1)
    player_rank: float = Field(7.0, description="playerankScore proxy for player quality")


class PlayerInput(BaseModel):
    """One player's attributes used to compute injury risk inside the pipeline."""
    height_cm:                         float = Field(181.0)
    weight_kg:                         float = Field(76.0)
    pace:                              float = Field(72.0)
    physic:                            float = Field(75.0)
    fifa_rating:                       float = Field(78.0)
    age:                               float = Field(26.0)
    bmi:                               float = Field(23.2)
    work_rate_numeric:                 float = Field(2.0,  description="Low=1, Medium=2, High=3")
    position_numeric:                  float = Field(3.0,  description="GK=1, DF=2, MF=3, FW=4")
    cumulative_minutes_played:         float = Field(8200.0)
    cumulative_games_played:           float = Field(95.0)
    cumulative_days_injured:           float = Field(0.0)
    minutes_per_game_prev_seasons:     float = Field(72.0)
    avg_days_injured_prev_seasons:     float = Field(0.0)
    avg_games_per_season_prev_seasons: float = Field(28.0)
    significant_injury_prev_season:    int   = Field(0,    ge=0, le=1,
                                                     description="1 if 28+ day injury last season")
    season_days_injured_prev_season:   float = Field(0.0)


class PipelinePredictionRequest(BaseModel):
    """
    Unified pipeline: runs xG → injury → match → adjustment in one call.

    - Provide `home_shots` / `away_shots` to enable xG analysis.
      Each shot needs at minimum X and Y; all other fields have sensible defaults.
    - Provide `home_players` / `away_players` to enable injury analysis.
      Each player needs at minimum `age` and any history fields you know;
      the rest default to league-average values.
    - Both lists are optional. If omitted, the pipeline runs the match model only
      (identical to model="match") and reports no adjustment factors.
    """
    model:        Literal["pipeline"]
    home_team_id: int            = Field(..., ge=0, le=43, example=24)
    away_team_id: int            = Field(..., ge=0, le=43, example=23)
    mw:           Optional[int]  = Field(None, ge=1, le=38, example=20)
    home_stats:   Optional[dict] = Field(None)
    away_stats:   Optional[dict] = Field(None)
    home_shots:   Optional[List[ShotInput]]   = Field(None,
                                description="Recent home team shots for xG aggregation (any number)")
    away_shots:   Optional[List[ShotInput]]   = Field(None,
                                description="Recent away team shots for xG aggregation")
    home_players: Optional[List[PlayerInput]] = Field(None,
                                description="Home squad players for injury risk assessment")
    away_players: Optional[List[PlayerInput]] = Field(None,
                                description="Away squad players for injury risk assessment")


PredictionRequest = Annotated[
    Union[
        MatchPredictionRequest,
        XGPredictionRequest,
        InjuryPredictionRequest,
        PipelinePredictionRequest,
    ],
    Field(discriminator="model"),
]


# ── Output schemas ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction:    str         = Field(..., description="Human-readable outcome label")
    probabilities: dict        = Field(..., description="Probability per class (final, after any adjustments)")
    confidence:    str         = Field(..., description="high / medium / low")
    model_used:    str         = Field(..., description="Identifier of the model that ran")
    cycle:         int         = Field(..., description="Which prediction cycle (1, 2, or 3)")
    # Cycle 2 only
    xg:            Optional[float] = Field(None, description="xG probability (Cycle 2 only)")
    distance_m:    Optional[float] = Field(None, description="Shot distance in metres (Cycle 2 only)")
    angle_deg:     Optional[float] = Field(None, description="Shot angle in degrees (Cycle 2 only)")
    # Pipeline mode only
    base_probabilities: Optional[dict] = Field(None,
                         description="Match model raw probabilities before xG/injury adjustment")
    xg_analysis:        Optional[dict] = Field(None,
                         description="Aggregated xG stats per team (pipeline mode only)")
    injury_analysis:    Optional[dict] = Field(None,
                         description="Aggregated injury risk per team (pipeline mode only)")
    pipeline_factors:   Optional[List[str]] = Field(None,
                         description="Plain-English explanation of each adjustment applied")
