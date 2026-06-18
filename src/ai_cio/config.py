"""Configuration for AI CIO v1."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_AI_CIO_REPORT = "cio:v1:report:{profile_id}"
CACHE_AI_CIO_OPINION = "cio:v1:opinion:{profile_id}"
CACHE_AI_CIO_RECOMMENDATIONS = "cio:v1:recommendations:{profile_id}"

CIO_OPINIONS = (
    "STRONG_BUY_PORTFOLIO",
    "ACCUMULATE",
    "MAINTAIN",
    "DEFENSIVE",
    "RECOVERY",
    "HIGH_RISK",
    "NO_ACTION",
)

RECOMMENDATION_CATEGORIES = (
    "PROFILE",
    "ALLOCATION",
    "STRATEGY",
    "RISK",
    "PORTFOLIO",
    "NO_ACTION",
)

PRIORITY_SURVIVAL = 1
PRIORITY_CAPITAL_PRESERVATION = 2
PRIORITY_GROWTH = 3

CIO_SCORE_WEIGHTS = {
    "executive_investment_score": 0.30,
    "opportunity_score": 0.20,
    "risk_score": 0.15,
    "confidence_score": 0.15,
    "capital_efficiency": 0.10,
    "executive_learning_score": 0.10,
}
