from __future__ import annotations
"""
Shared dependency-injection functions.

Import these with FastAPI's Depends() to access the model registry
and feature store without importing global state directly in routers.

Currently used implicitly (services call the store/registry directly),
but exposed here for future endpoints that may need fine-grained DI
(e.g. selecting a specific model version, swapping the feature store
for a test double in integration tests).
"""

from api.services.model_registry import get_entry, ModelEntry
from api.ml.feature_store import get_team_stats


def get_model(name: str) -> ModelEntry:
    """Return a loaded ModelEntry by name ('match', 'xg', 'injury')."""
    return get_entry(name)


def get_team_rolling_stats(team_id: int) -> dict | None:
    """Return latest rolling feature dict for a team, or None if not found."""
    return get_team_stats(team_id)
