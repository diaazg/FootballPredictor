"""
Model registry: loads all .pkl artefacts once at startup.
Provides a typed accessor used by all services.
"""

import os
import joblib
from dataclasses import dataclass

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

_CYCLE_META = {
    "match": {
        "description":  "Match outcome prediction (Home Win / Draw / Away Win)",
        "cycle":        1,
        "model_type":   "XGBoost (tuned, chronological)",
        "primary_metric": "accuracy",
        "primary_value":  0.5022,
        "auc":            None,
        "note": "50.22% test accuracy on chronological hold-out (last 20% of matches by date) "
                "vs 48.89% dummy baseline (Dataset 2, rolling features). The legacy 57.33% number "
                "was inflated by leaky-CV scaling and is preserved in models/random_split_backup/.",
        "model_file":   "cycle1_xgb_best.pkl",
        "scaler_file":  "cycle1_scaler.pkl",
        "features_file":"cycle1_feature_cols.pkl",
    },
    "xg": {
        "description":  "Expected Goals (xG) probability for a shot",
        "cycle":        2,
        "model_type":   "XGBoost (tuned, chronological)",
        "primary_metric": "auc",
        "primary_value":  0.8342,
        "auc":            0.8342,
        "note": "AUC=0.8342 (chronological matchId split) vs 0.8183 random split; 9:1 class imbalance",
        "model_file":   "cycle2_best_model.pkl",
        "scaler_file":  "cycle2_scaler.pkl",
        "features_file":"cycle2_feature_cols.pkl",
    },
    "injury": {
        "description":  "Player injury risk (High = 28+ days missed this season)",
        "cycle":        3,
        "model_type":   "XGBoost (tuned, chronological)",
        "primary_metric": "auc",
        "primary_value":  0.6723,
        "auc":            0.6723,
        "note": "AUC=0.6723 (chronological start_year split) vs 0.6220 random LR; above published 0.60-0.70 range",
        "model_file":   "cycle3_best_model.pkl",
        "scaler_file":  "cycle3_scaler.pkl",
        "features_file":"cycle3_feature_cols.pkl",
    },
}

# Comparison table: all models evaluated per cycle
_CYCLE_COMPARISONS = {
    "match": [
        {"name": "XGBoost (tuned, chronological)", "accuracy": 0.5022, "auc": None, "note": "Best — saved model (chronological hold-out)"},
        {"name": "Random Forest (untuned, chrono)","accuracy": 0.5111, "auc": None, "note": "Untuned RF on chronological split"},
        {"name": "XGBoost (untuned, chrono)",       "accuracy": 0.5022, "auc": None, "note": "Untuned XGB on chronological split"},
        {"name": "Logistic Regression (chrono)",    "accuracy": 0.4356, "auc": None, "note": "LR baseline on chronological split"},
        {"name": "XGBoost (tuned, random) [legacy]","accuracy": 0.5733, "auc": None, "note": "Legacy random-split with leaky CV (in random_split_backup/)"},
        {"name": "XGBoost (tuned, random) [honest]","accuracy": 0.4622, "auc": None, "note": "Random-split with proper Pipeline-based CV"},
        {"name": "Dummy (most_frequent)",          "accuracy": 0.4889, "auc": None, "note": "Chronological dummy floor"},
    ],
    "xg": [
        {"name": "XGBoost (tuned, chronological)",  "accuracy": 0.7244, "auc": 0.8342, "note": "Best — saved model (chronological matchId split)"},
        {"name": "Random Forest (tuned, chronological)", "accuracy": None, "auc": 0.8292, "note": "Chronological tuned RF"},
        {"name": "XGBoost (tuned, random)",         "accuracy": 0.7339, "auc": 0.8183, "note": "Random-split baseline (legacy)"},
        {"name": "Random Forest (tuned, random)",   "accuracy": 0.7179, "auc": 0.8176, "note": "Random-split (legacy)"},
        {"name": "Logistic Regression (random)",    "accuracy": 0.7250, "auc": 0.7963, "note": "Random-split baseline (legacy)"},
        {"name": "Dummy",                           "accuracy": 0.8918, "auc": 0.5000, "note": "Baseline floor"},
    ],
    "injury": [
        {"name": "XGBoost (tuned, chronological)",  "accuracy": 0.6736, "auc": 0.6723, "note": "Best — saved model (chronological start_year split)"},
        {"name": "Random Forest (tuned, chronological)", "accuracy": None, "auc": 0.6668, "note": "Chronological tuned RF"},
        {"name": "Logistic Regression (chronological)", "accuracy": None, "auc": 0.6263, "note": "LR baseline on chronological split"},
        {"name": "Logistic Regression (random)",    "accuracy": 0.5517, "auc": 0.6220, "note": "Previous deployed model (legacy)"},
        {"name": "XGBoost (tuned, random)",         "accuracy": 0.6552, "auc": 0.6179, "note": "Random-split tuned (legacy)"},
        {"name": "Random Forest (tuned, random)",   "accuracy": 0.6743, "auc": 0.6170, "note": "Random-split tuned (legacy)"},
        {"name": "Dummy",                           "accuracy": 0.7011, "auc": 0.5000, "note": "Baseline floor"},
    ],
}


@dataclass
class ModelEntry:
    model:        object
    scaler:       object
    feature_cols: list[str]
    meta:         dict


_registry: dict[str, ModelEntry] = {}
_ready: bool = False


def load_registry() -> None:
    global _ready
    for name, meta in _CYCLE_META.items():
        _registry[name] = ModelEntry(
            model        = joblib.load(os.path.join(_MODELS_DIR, meta["model_file"])),
            scaler       = joblib.load(os.path.join(_MODELS_DIR, meta["scaler_file"])),
            feature_cols = joblib.load(os.path.join(_MODELS_DIR, meta["features_file"])),
            meta         = meta,
        )
    _ready = True


def get_entry(name: str) -> ModelEntry:
    if name not in _registry:
        raise KeyError(f"Unknown model: '{name}'. Valid options: {list(_registry)}")
    return _registry[name]


def list_models() -> list[str]:
    return list(_registry.keys())


def get_comparisons(name: str) -> list[dict]:
    return _CYCLE_COMPARISONS.get(name, [])


def is_ready() -> bool:
    return _ready
