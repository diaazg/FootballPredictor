"""
Shared path configuration for all notebooks and scripts.

Usage (add to the first code cell of any notebook):

    import sys, os
    _here = os.getcwd()
    while not os.path.isdir(os.path.join(_here, 'data')):
        _p = os.path.dirname(_here)
        if _p == _here: raise RuntimeError("project root not found")
        _here = _p
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from config import Paths

Then use Paths.XYZ wherever a file path is needed.
"""

from pathlib import Path

ROOT = Path(__file__).parent


class Paths:
    # ── Project root ──────────────────────────────────────────────────────────
    ROOT = ROOT

    # ── Data directories ──────────────────────────────────────────────────────
    DATA_RAW       = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"

    # ── Raw files ─────────────────────────────────────────────────────────────
    PL_MATCHES_RAW        = DATA_RAW / "premier_league_matches.csv"
    EVENTS_ENGLAND        = DATA_RAW / "events_England.json"
    PLAYERANK             = DATA_RAW / "playerank.json"
    PLAYERS               = DATA_RAW / "players.json"
    TAGS2NAME             = DATA_RAW / "tags2name.csv"
    EVENTID2NAME          = DATA_RAW / "eventid2name.csv"
    PLAYER_INJURIES_RAW   = DATA_RAW / "player_injuries.csv"

    # ── Processed files ───────────────────────────────────────────────────────
    PL_MATCHES_PROCESSED      = DATA_PROCESSED / "premier_league_matches_processed.csv"
    WYSCOUT_PROCESSED         = DATA_PROCESSED / "wyscout_shots_processed.csv"
    PLAYER_INJURIES_PROCESSED = DATA_PROCESSED / "player_injuries_processed.csv"

    # ── Models root ───────────────────────────────────────────────────────────
    MODELS    = ROOT / "models"
    MODELS_C1 = MODELS / "cycle1"
    MODELS_C2 = MODELS / "cycle2"
    MODELS_C3 = MODELS / "cycle3"

    # ── Cycle 1 model artefacts ───────────────────────────────────────────────
    C1_MODEL    = MODELS_C1 / "cycle1_xgb_best.pkl"
    C1_FEATURES = MODELS_C1 / "cycle1_feature_cols.pkl"

    # ── Cycle 2 model artefacts ───────────────────────────────────────────────
    C2_MODEL    = MODELS_C2 / "cycle2_best_model.pkl"
    C2_SCALER   = MODELS_C2 / "cycle2_scaler.pkl"
    C2_FEATURES = MODELS_C2 / "cycle2_feature_cols.pkl"

    # ── Cycle 3 model artefacts ───────────────────────────────────────────────
    C3_MODEL    = MODELS_C3 / "cycle3_best_model.pkl"
    C3_SCALER   = MODELS_C3 / "cycle3_scaler.pkl"
    C3_FEATURES = MODELS_C3 / "cycle3_feature_cols.pkl"


# Convenience: ensure all model subdirs exist when this module is imported
def ensure_dirs() -> None:
    for d in [Paths.MODELS_C1, Paths.MODELS_C2, Paths.MODELS_C3]:
        d.mkdir(parents=True, exist_ok=True)
