from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.schemas.model_info import ModelInfo, ModelCompareResponse, ModelVariant
from api.services.model_registry import get_entry, list_models, get_comparisons

router = APIRouter()


@router.get(
    "/models",
    response_model=list[ModelInfo],
    summary="List all available prediction models",
)
def models_list():
    """Return metadata for all three cycle models currently loaded in the registry."""
    result = []
    for name in list_models():
        entry = get_entry(name)
        m = entry.meta
        result.append(ModelInfo(
            name           = name,
            description    = m["description"],
            cycle          = m["cycle"],
            model_type     = m["model_type"],
            primary_metric = m["primary_metric"],
            primary_value  = m["primary_value"],
            auc            = m["auc"],
            note           = m["note"],
            feature_count  = len(entry.feature_cols),
        ))
    return result


@router.get(
    "/models/{name}/compare",
    response_model=ModelCompareResponse,
    summary="Compare all evaluated models for one cycle",
)
def models_compare(name: str):
    """
    Return a side-by-side comparison of every model evaluated during the
    specified cycle, including tuned and untuned variants.

    `name` must be one of: `match`, `xg`, `injury`.
    """
    if name not in list_models():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{name}'. Valid options: {list_models()}",
        )
    entry     = get_entry(name)
    variants  = get_comparisons(name)
    best_name = entry.meta["model_type"]

    return ModelCompareResponse(
        cycle    = entry.meta["cycle"],
        best     = best_name,
        variants = [ModelVariant(**v) for v in variants],
    )
