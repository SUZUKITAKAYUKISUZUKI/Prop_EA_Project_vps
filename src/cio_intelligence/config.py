"""Configuration for CIO Intelligence Layer v1."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_CIO_INTELLIGENCE = "cio:intelligence:{profile_id}"
CACHE_CIO_STATE = "cio:state:{profile_id}"
CACHE_CIO_OPPORTUNITY = "cio:opportunity:{profile_id}"
CACHE_CIO_RISK = "cio:risk:{profile_id}"

INVESTMENT_STATES = (
    "UNDER_ALLOCATED",
    "BALANCED",
    "OVER_ALLOCATED",
    "HIGH_GROWTH",
    "HIGH_RISK",
    "RECOVERY_MODE",
    "CAPITAL_EFFICIENT",
    "STRATEGY_CONCENTRATED",
    "DIVERSIFIED",
)

EXECUTIVE_INVESTMENT_WEIGHTS = {
    "portfolio_health": 0.25,
    "capital_efficiency": 0.20,
    "opportunity_score": 0.20,
    "risk_score": 0.15,
    "confidence_score": 0.10,
    "intelligence_trust": 0.10,
}
