"""Configuration for Autonomous Portfolio Manager v1."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_APM_STATUS = "apm:status:{profile_id}"
CACHE_APM_QUEUE = "apm:queue:{profile_id}"
CACHE_APM_ROADMAP = "apm:roadmap:{profile_id}"
CACHE_APM_OPPORTUNITIES = "apm:opportunities:{profile_id}"
CACHE_APM_RISK_ALERTS = "apm:risk_alerts:{profile_id}"

EXECUTIVE_ACTIONS = (
    "PROFILE_SWITCH",
    "ALLOCATION_REBALANCE",
    "PROMOTE_STRATEGY",
    "DEMOTE_STRATEGY",
    "RETIRE_STRATEGY",
    "ENTER_RECOVERY",
    "EXIT_RECOVERY",
    "NO_ACTION",
)

APPROVAL_STATUSES = ("PENDING_APPROVAL", "APPROVED", "REJECTED", "EXECUTED")

EXECUTIVE_WEIGHTS = {
    "governor_confidence": 0.30,
    "portfolio_health": 0.25,
    "risk_budget": 0.20,
    "trust_score": 0.15,
    "consensus": 0.10,
}

EXECUTIVE_CATEGORIES: tuple[tuple[float, str], ...] = (
    (90.0, "EXCELLENT"),
    (75.0, "STRONG"),
    (60.0, "MODERATE"),
    (45.0, "WEAK"),
    (0.0, "CRITICAL"),
)


def executive_category(score: float) -> str:
    for threshold, label in EXECUTIVE_CATEGORIES:
        if score >= threshold:
            return label
    return "CRITICAL"
