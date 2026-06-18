"""Configuration for Operational Readiness Layer v1."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_ORL_READINESS = "orl:readiness:{profile_id}"
CACHE_ORL_HEALTH = "orl:health:{profile_id}"

READINESS_WEIGHTS = {
    "system_health": 0.20,
    "database_health": 0.15,
    "api_health": 0.15,
    "dashboard_health": 0.15,
    "cache_health": 0.10,
    "recommendation_consistency": 0.10,
    "ai_cio_availability": 0.10,
    "historical_stability": 0.05,
}

READINESS_THRESHOLDS = (
    (95, "PRODUCTION_READY"),
    (85, "READY"),
    (70, "ACCEPTABLE"),
    (50, "WARNING"),
    (0, "NOT_READY"),
)

REQUIRED_DB_TABLES = (
    "cio_reports",
    "cio_intelligence_snapshots",
    "cio_recommendations",
    "executive_investment_scores",
    "apm_execution_queue",
    "analytics_cache",
    "operational_readiness",
    "operational_audit_log",
    "system_health_history",
)
