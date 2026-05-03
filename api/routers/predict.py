from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.services import prediction_service as svc

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run a prediction (all three cycles via a single endpoint)",
)
def predict(body: PredictionRequest):
    """
    Unified prediction endpoint for all three Football Predictor cycles.

    Set the **`model`** field to select the prediction type:

    - `"match"` — Predict match outcome (Home Win / Draw / Away Win).
      Provide `home_team_id`, `away_team_id`, and optionally `mw` (matchweek 1-38).
      Each team's season-to-date stats are looked up automatically from the feature store.

    - `"xg"` — Predict Expected Goals probability for a shot.
      Provide raw X/Y coordinates (0-100 scale). Distance and Angle are
      computed automatically.

    - `"injury"` — Predict player injury risk (High = 28+ days missed).
      Provide all 17 player attributes and injury history fields.

    - `"pipeline"` — **Integrated mode.** Runs all three models in sequence and
      combines their outputs into one response.
      - Provide `home_team_id` and `away_team_id` (required).
      - Optionally provide `home_shots` / `away_shots` (list of shot objects) to
        enable xG analysis. The xG model runs on each shot; results are aggregated
        into `total_xg`, `avg_xg_per_shot`, and `shot_count` per team.
      - Optionally provide `home_players` / `away_players` (list of player objects)
        to enable injury analysis. The injury model runs on each player; results
        are aggregated into `high_risk_count` and `avg_risk_probability` per team.
      - The match model produces **base probabilities**. The xG and injury
        aggregates then **adjust** those probabilities (up to ±5% each factor)
        to produce the final `probabilities` in the response.
      - `pipeline_factors` in the response explains in plain English what adjustments
        were made and why.

    Returns `prediction`, `probabilities`, `confidence` (high/medium/low),
    and `model_used` for all types. Cycle 2 also returns `xg`, `distance_m`,
    and `angle_deg`. Pipeline mode also returns `base_probabilities`,
    `xg_analysis`, `injury_analysis`, and `pipeline_factors`.
    """
    try:
        if body.model == "match":
            result = svc.predict_match(
                body.home_team_id,
                body.away_team_id,
                mw=body.mw,
                home_stats=body.home_stats,
                away_stats=body.away_stats,
            )
        elif body.model == "xg":
            result = svc.predict_xg(
                body.X, body.Y,
                body.left_foot, body.right_foot, body.header,
                body.first_half, body.player_rank,
            )
        elif body.model == "pipeline":
            result = svc.predict_pipeline(
                home_team_id  = body.home_team_id,
                away_team_id  = body.away_team_id,
                mw            = body.mw,
                home_stats    = body.home_stats,
                away_stats    = body.away_stats,
                home_shots    = body.home_shots,
                away_shots    = body.away_shots,
                home_players  = body.home_players,
                away_players  = body.away_players,
            )
        else:
            features = body.model_dump(exclude={"model"})
            result = svc.predict_injury(features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
