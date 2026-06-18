"""Configuration for RC2 Live Operations Layer."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_RC2_BRIEFING = "rc2:briefing:{profile_id}:{date}"
CACHE_RC2_DIGEST = "rc2:digest:{profile_id}:{date}"
CACHE_RC2_OPERATIONS = "rc2:operations:{profile_id}"

OPERATIONAL_SCORE_WEIGHTS = {
    "ai_cio_availability": 0.25,
    "system_health": 0.20,
    "recommendation_stability": 0.20,
    "data_integrity": 0.15,
    "user_action_load": 0.10,
    "historical_reliability": 0.10,
}

READINESS_THRESHOLDS = (
    (95, "PRODUCTION_READY"),
    (90, "READY"),
    (75, "ACCEPTABLE"),
    (60, "WARNING"),
    (0, "NOT_READY"),
)

USER_ACTION_LOAD_IDEAL_MAX = 3
USER_ACTION_LOAD_WARNING = 10

NOTIFICATION_LEVELS = ("INFO", "NOTICE", "WARNING", "CRITICAL")
