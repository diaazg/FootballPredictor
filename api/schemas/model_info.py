from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name:             str        = Field(..., description="Model identifier used in /predict")
    description:      str        = Field(..., description="What this model predicts")
    cycle:            int        = Field(..., description="Prediction cycle (1, 2, or 3)")
    model_type:       str        = Field(..., description="Algorithm used")
    primary_metric:   str        = Field(..., description="Primary evaluation metric")
    primary_value:    float      = Field(..., description="Primary metric value on test set")
    auc:              Optional[float] = Field(None, description="AUC-ROC (None for Cycle 1)")
    note:             str        = Field(..., description="Context and caveats")
    feature_count:    int        = Field(..., description="Number of input features")


class ModelVariant(BaseModel):
    name:      str        = Field(..., description="Model variant name")
    accuracy:  Optional[float] = Field(None)
    auc:       Optional[float] = Field(None)
    note:      str        = Field(...)


class ModelCompareResponse(BaseModel):
    cycle:     int                = Field(..., description="Which cycle these models belong to")
    best:      str                = Field(..., description="Name of the saved (best) model")
    variants:  list[ModelVariant] = Field(..., description="All evaluated variants for this cycle")
