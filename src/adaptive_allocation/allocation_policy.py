"""Profile- and state-aware allocation policies for PAAE."""
from __future__ import annotations

from typing import Any

from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES

STATE_OBJECTIVES: dict[str, dict[str, Any]] = {
    "challenge": {
        "label": "FASTEST PASS",
        "priorities": ("pass_rate", "pass_days"),
        "base_weights": {
            "LSFC": 0.50,
            "DBBS": 0.30,
            "DiNapoli": 0.15,
            "VAMR": 0.03,
            "SMRS": 0.02,
        },
    },
    "funded": {
        "label": "CAPITAL GROWTH",
        "priorities": ("total_r", "pf", "sharpe"),
        "base_weights": {
            "LSFC": 0.40,
            "DBBS": 0.25,
            "DiNapoli": 0.25,
            "VAMR": 0.05,
            "SMRS": 0.05,
        },
    },
    "recovery": {
        "label": "DD REDUCTION",
        "priorities": ("risk_score",),
        "base_weights": {
            "LSFC": 0.50,
            "DBBS": 0.30,
            "DiNapoli": 0.20,
            "VAMR": 0.00,
            "SMRS": 0.00,
        },
    },
    "live": {
        "label": "CAPITAL GROWTH",
        "priorities": ("total_r", "pf", "sharpe"),
        "base_weights": {
            "LSFC": 0.35,
            "DBBS": 0.25,
            "DiNapoli": 0.25,
            "VAMR": 0.10,
            "SMRS": 0.05,
        },
    },
}


def resolve_account_state(profile_id: str, settings: dict[str, str] | None = None) -> str:
    settings = settings or {}
    objective = str(settings.get("recommended_objective", "")).upper()
    profile_key = str(settings.get("profile_key", "")).lower()
    if "Challenge" in profile_id or profile_key == "challenge":
        return "challenge"
    if "Recovery" in profile_id:
        return "recovery"
    if profile_key == "live" or "Live" in profile_id:
        return "live"
    if profile_key == "funded" or "Funded" in profile_id:
        return "funded"
    if "PRESERVATION" in objective or "DD REDUCTION" in objective:
        return "recovery"
    if "FASTEST" in objective:
        return "challenge"
    return "funded"


def base_weights_for_state(account_state: str) -> dict[str, float]:
    policy = STATE_OBJECTIVES.get(account_state.lower(), STATE_OBJECTIVES["funded"])
    base = dict(policy["base_weights"])
    for code in DASHBOARD_STRATEGY_CODES:
        base.setdefault(code, 0.0)
    total = sum(base.values()) or 1.0
    return {k: round(v / total, 4) for k, v in base.items()}


def policy_metadata(account_state: str) -> dict[str, Any]:
    policy = STATE_OBJECTIVES.get(account_state.lower(), STATE_OBJECTIVES["funded"])
    return {
        "account_state": account_state,
        "objective": policy["label"],
        "priorities": list(policy["priorities"]),
        "base_weights": dict(policy["base_weights"]),
    }
