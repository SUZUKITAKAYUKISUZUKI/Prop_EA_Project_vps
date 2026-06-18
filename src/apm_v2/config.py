"""Configuration for APM v2 executive memory layer."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_V2_MEMORY = "apm:v2:memory:{profile_id}"
CACHE_V2_LESSONS = "apm:v2:lessons:{profile_id}"
CACHE_V2_BOARD = "apm:v2:board:{profile_id}"
CACHE_V2_SCORE = "apm:v2:score:{profile_id}"
CACHE_V2_INTELLIGENCE = "apm:v2:intelligence:{profile_id}"

DIRECTORS = (
    "RISK_DIRECTOR",
    "GROWTH_DIRECTOR",
    "STRATEGY_DIRECTOR",
    "CONFIDENCE_DIRECTOR",
    "INTELLIGENCE_DIRECTOR",
)

EFFECTIVENESS_CATEGORIES = (
    "PROFILE_SWITCH",
    "ALLOCATION",
    "PROMOTION",
    "RECOVERY",
    "RETIREMENT",
)

EXECUTIVE_V2_WEIGHTS = {
    "decision_accuracy": 0.35,
    "learning_quality": 0.25,
    "portfolio_improvement": 0.20,
    "governor_reliability": 0.10,
    "confidence_reliability": 0.10,
}

OUTCOME_SUCCESS_THRESHOLD = 60.0
