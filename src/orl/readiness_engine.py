"""Operational readiness score for ORL v1."""
from __future__ import annotations

from typing import Any

from src.orl.config import READINESS_THRESHOLDS, READINESS_WEIGHTS


class ReadinessEngine:
    def evaluate(self, components: dict[str, Any]) -> dict[str, Any]:
        readiness_score = round(
            READINESS_WEIGHTS["system_health"] * float(components.get("system_health") or 0)
            + READINESS_WEIGHTS["database_health"] * float(components.get("database_health") or 0)
            + READINESS_WEIGHTS["api_health"] * float(components.get("api_health") or 0)
            + READINESS_WEIGHTS["dashboard_health"] * float(components.get("dashboard_health") or 0)
            + READINESS_WEIGHTS["cache_health"] * float(components.get("cache_health") or 0)
            + READINESS_WEIGHTS["recommendation_consistency"] * float(components.get("recommendation_consistency") or 0)
            + READINESS_WEIGHTS["ai_cio_availability"] * float(components.get("ai_cio_availability") or 0)
            + READINESS_WEIGHTS["historical_stability"] * float(components.get("historical_stability") or 0),
            2,
        )

        status = self._status(readiness_score)
        production_ready = readiness_score >= 85 and not components.get("critical_issues")

        return {
            "readiness_score": readiness_score,
            "readiness_status": status,
            "production_ready": production_ready,
            "readiness_components": {
                "system_health": components.get("system_health"),
                "database_health": components.get("database_health"),
                "api_health": components.get("api_health"),
                "dashboard_health": components.get("dashboard_health"),
                "cache_health": components.get("cache_health"),
                "recommendation_consistency": components.get("recommendation_consistency"),
                "ai_cio_availability": components.get("ai_cio_availability"),
                "historical_stability": components.get("historical_stability"),
            },
        }

    def _status(self, score: float) -> str:
        for threshold, label in READINESS_THRESHOLDS:
            if score >= threshold:
                return label
        return "NOT_READY"
