"""Configuration for Portfolio OS RC1 production hardening."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_PRODUCTION_READINESS = "production:readiness:{profile_id}"
CACHE_PRODUCTION_BENCHMARKS = "production:benchmarks:{profile_id}"

RESILIENCE_WEIGHTS = {
    "data_integrity": 0.25,
    "api_availability": 0.20,
    "dashboard_availability": 0.20,
    "recommendation_consistency": 0.15,
    "ai_cio_availability": 0.10,
    "failure_recovery": 0.10,
}

READINESS_THRESHOLDS = (
    (95, "PRODUCTION_READY"),
    (85, "READY"),
    (70, "ACCEPTABLE"),
    (50, "WARNING"),
    (0, "NOT_READY"),
)

CHAIN_LAYERS = (
    "prae",
    "paae",
    "pdts",
    "slm",
    "age",
    "cace",
    "mie",
    "apm",
    "cil",
    "ai_cio",
    "orl",
)

RECOMMENDATION_CHAIN = ("ai_cio", "cil", "apm", "age", "paae")

REQUIRED_DB_TABLES = (
    "cio_reports",
    "cio_intelligence_snapshots",
    "operational_readiness",
    "production_readiness",
    "production_validation_history",
    "production_benchmark_history",
    "production_failures",
    "production_resilience_history",
    "analytics_cache",
)
