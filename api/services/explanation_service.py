from __future__ import annotations
"""
SHAP explanation service.

Uses TreeExplainer for XGBoost models (Cycles 1 & 2)
and LinearExplainer for Logistic Regression (Cycle 3).
"""

import numpy as np
import pandas as pd
import shap
from api.services.model_registry import get_entry
from api.ml.pipeline import build_match_vector, build_xg_vector, build_injury_vector


_TREE_MODELS = {
    "XGBClassifier",
    "LGBMClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "CatBoostClassifier",
    "DecisionTreeClassifier",
}


def _explain_single(model_name: str, X_raw: pd.DataFrame, top_n: int) -> dict:
    entry = get_entry(model_name)
    X_scaled = entry.scaler.transform(X_raw) if entry.scaler is not None else X_raw.values

    model_type = type(entry.model).__name__

    if model_type in _TREE_MODELS:
        explainer = shap.TreeExplainer(entry.model)
        shap_values = explainer.shap_values(X_scaled)
        ev = explainer.expected_value

        if np.ndim(shap_values) == 3:
            # Modern shap: (n_samples, n_features, n_classes)
            proba = entry.model.predict_proba(X_scaled)[0]
            cls = int(np.argmax(proba))
            sv = shap_values[0, :, cls]
            base_value = float(ev[cls]) if hasattr(ev, "__len__") else float(ev)
        elif isinstance(shap_values, list):
            # Legacy shap: list of (n_samples, n_features) per class
            proba = entry.model.predict_proba(X_scaled)[0]
            cls = int(np.argmax(proba))
            sv = shap_values[cls][0]
            base_value = float(ev[cls]) if hasattr(ev, "__len__") else float(ev)
        else:
            # Binary: 2-D (n_samples, n_features)
            sv = shap_values[0]
            base_value = float(ev)
    else:
        # LinearExplainer for LogisticRegression.
        # Background = zeros = training mean in standardized space.
        background = np.zeros((1, X_scaled.shape[1]))
        explainer = shap.LinearExplainer(entry.model, background)
        shap_values = explainer.shap_values(X_scaled)
        ev = explainer.expected_value

        if isinstance(shap_values, list):
            proba = entry.model.predict_proba(X_scaled)[0]
            cls = int(np.argmax(proba))
            sv = shap_values[cls][0] if len(shap_values) > 1 else shap_values[0][0]
            base_value = float(ev[cls]) if hasattr(ev, "__len__") else float(ev)
        else:
            sv = shap_values[0]
            base_value = float(ev) if not hasattr(ev, "__len__") else float(ev[0])

    feature_names = entry.feature_cols
    raw_values = X_raw.iloc[0].to_dict()

    impacts = [
        {
            "feature":    name,
            "value":      round(float(raw_values[name]), 4),
            "shap_value": round(float(sv[i]), 4),
            "impact":     "increases_risk" if sv[i] > 0 else "decreases_risk",
        }
        for i, name in enumerate(feature_names)
    ]
    impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "base_value":    round(base_value, 4),
        "top_features":  impacts[:top_n],
        "all_features":  impacts,
    }


def explain_match(
    home_team_id: int,
    away_team_id: int,
    top_n: int = 10,
    mw: int | None = None,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
) -> dict:
    entry = get_entry("match")
    X = build_match_vector(home_team_id, away_team_id, entry.feature_cols,
                           mw=mw,
                           home_stats_override=home_stats,
                           away_stats_override=away_stats)
    return _explain_single("match", X, top_n)


def explain_xg(
    X: float,
    Y: float,
    left_foot: int,
    right_foot: int,
    header: int,
    first_half: int,
    player_rank: float,
    top_n: int = 9,
) -> dict:
    entry = get_entry("xg")
    feat_df, _, _ = build_xg_vector(
        X, Y, left_foot, right_foot, header, first_half, player_rank,
        entry.feature_cols,
    )
    return _explain_single("xg", feat_df, top_n)


def explain_injury(features: dict, top_n: int = 10) -> dict:
    entry = get_entry("injury")
    X = build_injury_vector(features, entry.feature_cols)
    return _explain_single("injury", X, top_n)
