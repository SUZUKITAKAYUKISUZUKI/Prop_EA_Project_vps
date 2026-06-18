"""Internal helpers for ORL v1."""
from __future__ import annotations

from typing import Any


def extract_health(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": report.get("profile_id"),
        "system_health_score": report.get("system_health_score"),
        "database_health": report.get("database_health"),
        "api_health": report.get("api_health"),
        "dashboard_health": report.get("dashboard_health"),
        "cache_health": report.get("cache_health"),
        "ai_cio_availability": report.get("ai_cio_availability"),
        "healthy": float(report.get("readiness_score") or 0) >= 85,
    }


def extract_audit(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": report.get("profile_id"),
        "audit": report.get("audit"),
        "open_operational_issues": report.get("open_operational_issues"),
        "critical_issues": (report.get("executive_summary") or {}).get("critical_issues"),
    }
