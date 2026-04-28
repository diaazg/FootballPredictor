from __future__ import annotations
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from api.schemas.prediction import (
    MatchPredictionRequest,
    XGPredictionRequest,
    InjuryPredictionRequest,
)


class MatchExplainRequest(MatchPredictionRequest):
    model: Literal["match"]  # type: ignore[assignment]
    top_n: int = Field(10, ge=1, le=33,
                       description="Number of top SHAP features to return")


class XGExplainRequest(XGPredictionRequest):
    model: Literal["xg"]  # type: ignore[assignment]
    top_n: int = Field(9, ge=1, le=9,
                       description="Number of top SHAP features to return")


class InjuryExplainRequest(InjuryPredictionRequest):
    model: Literal["injury"]  # type: ignore[assignment]
    top_n: int = Field(10, ge=1, le=17,
                       description="Number of top SHAP features to return")


ExplainRequest = Annotated[
    Union[MatchExplainRequest, XGExplainRequest, InjuryExplainRequest],
    Field(discriminator="model"),
]


class FeatureImpact(BaseModel):
    feature:    str   = Field(..., description="Feature name")
    value:      float = Field(..., description="Input value for this prediction")
    shap_value: float = Field(..., description="SHAP contribution (positive = pushes prediction up)")
    impact:     str   = Field(..., description="increases_risk or decreases_risk")


class ExplainResponse(BaseModel):
    base_value:   float               = Field(..., description="Model expected output before seeing features")
    top_features: list[FeatureImpact] = Field(..., description="Most impactful features (sorted by |SHAP|)")
