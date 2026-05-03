from __future__ import annotations
"""
Football Predictor API — app factory.

Startup sequence (lifespan):
  1. Load all models/*.pkl  → model_registry dict
  2. Load skysports_match_stats_cleaned.csv
     → compute latest 5-match rolling window per team → feature_store dict
  3. /health/ready returns 200 OK
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.services.model_registry import load_registry
from api.ml.feature_store import build_feature_store
from api.routers import health, predict, explain, models, teams


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    load_registry()          # loads cycle1/2/3 .pkl files
    build_feature_store()    # computes rolling team stats from cleaned CSV
    yield
    # ── Shutdown (nothing to release) ────────────────────────────────────────


app = FastAPI(
    title="Football Analaytics AI API",
    description=(
        "Three prediction cycles served through a unified interface:\n\n"
        "| Model key | Cycle | Task | Algorithm | Best metric |\n"
        "|-----------|-------|------|-----------|-------------|\n"
        "| `match`   | 1 | W/D/L match outcome        | XGBoost (tuned, chronological) | 52.85% accuracy |\n"
        "| `xg`      | 2 | Expected Goals probability | XGBoost (tuned, random)        | AUC 0.8183 |\n"
        "| `injury`  | 3 | Player injury risk         | LightGBM (tuned, chronological) | AUC 0.68 |\n\n"
        "All predictions go through **POST /predict** using a `model` discriminator field. "
        "SHAP explanations are available via **POST /explain**.\n\n"
        "Interactive docs: `/docs` — ReDoc: `/redoc`"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(health.router,  tags=["Health"])
app.include_router(predict.router, tags=["Predictions"])
app.include_router(explain.router, tags=["Explanations"])
app.include_router(models.router,  tags=["Model Info"])
app.include_router(teams.router,   tags=["Teams"])
