"""Operational report builder for ORL v1."""
from __future__ import annotations

from typing import Any


class OperationalReport:
    def build(
        self,
        *,
        profile_id: str,
        readiness: dict[str, Any],
        system_health: dict[str, Any],
        database: dict[str, Any],
        api: dict[str, Any],
        dashboard: dict[str, Any],
        cache: dict[str, Any],
        consistency: dict[str, Any],
        audit: dict[str, Any],
        ai_cio: dict[str, Any],
        executive: dict[str, Any],
        historical_stability: float,
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            **readiness,
            "system_health_score": system_health.get("system_health"),
            "database_health": database.get("database_health"),
            "api_health": api.get("api_health"),
            "dashboard_health": dashboard.get("dashboard_health"),
            "cache_health": cache.get("cache_health"),
            "recommendation_consistency": consistency.get("recommendation_consistency"),
            "ai_cio_availability": ai_cio.get("ai_cio_availability"),
            "historical_stability": historical_stability,
            "system_health": system_health,
            "database": database,
            "api": api,
            "dashboard": dashboard,
            "cache": cache,
            "consistency": consistency,
            "audit": audit,
            "ai_cio_check": ai_cio,
            "executive_summary": executive,
            "open_operational_issues": executive.get("open_operational_issues"),
            "can_operate_with_ai_cio_alone": executive.get("can_operate_with_ai_cio_alone"),
        }
