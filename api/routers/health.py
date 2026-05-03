from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.services.model_registry import is_ready as models_ready
from api.ml.feature_store import is_ready as store_ready

router = APIRouter()


@router.get("/health", summary="Basic liveness probe")
def health():
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe — all models and feature store loaded")
def ready():
    if not models_ready():
        raise HTTPException(status_code=503, detail="Model registry not loaded yet.")
    if not store_ready():
        raise HTTPException(status_code=503, detail="Feature store not loaded yet.")
    return {"status": "ready", "models": True, "feature_store": True}
