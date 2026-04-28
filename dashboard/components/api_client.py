"""
Thin wrapper around the Football Predictor FastAPI.
All dashboard pages import from here — one place to change the base URL.
"""
from __future__ import annotations
import requests

BASE_URL = "http://localhost:8000"
TIMEOUT  = 10


def _post(path: str, payload: dict) -> dict:
    r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _get(path: str) -> dict | list:
    r = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Predictions ───────────────────────────────────────────────────────────────

def predict_match(
    home_team_id: int,
    away_team_id: int,
    mw: int | None = None,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
) -> dict:
    payload = {
        "model": "match",
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
    }
    if mw is not None:
        payload["mw"] = mw
    if home_stats is not None:
        payload["home_stats"] = home_stats
    if away_stats is not None:
        payload["away_stats"] = away_stats
    return _post("/predict", payload)


def predict_xg(X, Y, left_foot, right_foot, header, first_half, player_rank) -> dict:
    return _post("/predict", {
        "model": "xg",
        "X": X, "Y": Y,
        "left_foot": left_foot,
        "right_foot": right_foot,
        "header": header,
        "first_half": first_half,
        "player_rank": player_rank,
    })


def predict_injury(features: dict) -> dict:
    return _post("/predict", {"model": "injury", **features})


# ── Explanations ──────────────────────────────────────────────────────────────

def explain_match(
    home_team_id: int,
    away_team_id: int,
    top_n: int = 10,
    mw: int | None = None,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
) -> dict:
    payload = {
        "model": "match",
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "top_n": top_n,
    }
    if mw is not None:
        payload["mw"] = mw
    if home_stats is not None:
        payload["home_stats"] = home_stats
    if away_stats is not None:
        payload["away_stats"] = away_stats
    return _post("/explain", payload)


def explain_xg(X, Y, left_foot, right_foot, header, first_half, player_rank, top_n: int = 9) -> dict:
    return _post("/explain", {
        "model": "xg",
        "X": X, "Y": Y,
        "left_foot": left_foot, "right_foot": right_foot,
        "header": header, "first_half": first_half,
        "player_rank": player_rank,
        "top_n": top_n,
    })


def explain_injury(features: dict, top_n: int = 10) -> dict:
    return _post("/explain", {"model": "injury", "top_n": top_n, **features})


# ── Model info ────────────────────────────────────────────────────────────────

def get_models() -> list:
    return _get("/models")


def get_model_compare(name: str) -> dict:
    return _get(f"/models/{name}/compare")


# ── Teams ─────────────────────────────────────────────────────────────────────

def get_teams() -> dict:
    return _get("/teams")


def get_team(team_id: int) -> dict:
    return _get(f"/teams/{team_id}")


# ── Health ────────────────────────────────────────────────────────────────────

def is_api_ready() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health/ready", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
