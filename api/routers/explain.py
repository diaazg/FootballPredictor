from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.schemas.explanation import ExplainRequest, ExplainResponse
from api.services import explanation_service as svc

router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="SHAP feature importance for a prediction",
)
def explain(body: ExplainRequest):
    """
    Return SHAP feature importance values for a given prediction input.

    Uses **TreeExplainer** for XGBoost models (Cycles 1 & 2) and
    **LinearExplainer** for Logistic Regression (Cycle 3).

    Set the `model` field exactly as in `/predict`. The `top_n` field
    (optional) controls how many features appear in `top_features`;
    `all_features` always contains the full ranked list.

    Each feature entry includes:
    - `feature` — feature name
    - `value` — the actual value sent in the request
    - `shap_value` — SHAP contribution (positive = pushes prediction toward positive class)
    - `impact` — `increases_risk` or `decreases_risk`
    """
    try:
        if body.model == "match":
            result = svc.explain_match(
                body.home_team_id,
                body.away_team_id,
                body.top_n,
                mw=body.mw,
                home_stats=body.home_stats,
                away_stats=body.away_stats,
            )
        elif body.model == "xg":
            result = svc.explain_xg(
                body.X, body.Y,
                body.left_foot, body.right_foot, body.header,
                body.first_half, body.player_rank,
                body.top_n,
            )
        else:
            features = body.model_dump(exclude={"model", "top_n"})
            result = svc.explain_injury(features, body.top_n)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
