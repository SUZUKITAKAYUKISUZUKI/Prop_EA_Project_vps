"""Resilience scoring for RC1."""
from __future__ import annotations

from typing import Any

from src.production_hardening.config import READINESS_THRESHOLDS, RESILIENCE_WEIGHTS


class ResilienceTester:
    def evaluate(self, components: dict[str, Any]) -> dict[str, Any]:
        resilience_score = round(
            RESILIENCE_WEIGHTS["data_integrity"] * float(components.get("data_integrity") or 0)
            + RESILIENCE_WEIGHTS["api_availability"] * float(components.get("api_availability") or 0)
            + RESILIENCE_WEIGHTS["dashboard_availability"] * float(components.get("dashboard_availability") or 0)
            + RESILIENCE_WEIGHTS["recommendation_consistency"] * float(components.get("recommendation_consistency") or 0)
            + RESILIENCE_WEIGHTS["ai_cio_availability"] * float(components.get("ai_cio_availability") or 0)
            + RESILIENCE_WEIGHTS["failure_recovery"] * float(components.get("failure_recovery") or 0),
            2,
        )
        status = self._status(resilience_score)
        return {
            "resilience_score": resilience_score,
            "resilience_status": status,
            "resilience_components": {
                "data_integrity": components.get("data_integrity"),
                "api_availability": components.get("api_availability"),
                "dashboard_availability": components.get("dashboard_availability"),
                "recommendation_consistency": components.get("recommendation_consistency"),
                "ai_cio_availability": components.get("ai_cio_availability"),
                "failure_recovery": components.get("failure_recovery"),
            },
            "production_ready": resilience_score >= 85,
        }

    def _status(self, score: float) -> str:
        for threshold, label in READINESS_THRESHOLDS:
            if score >= threshold:
                return label
        return "NOT_READY"
