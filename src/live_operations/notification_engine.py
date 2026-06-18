"""Notification routing for RC2 live operations."""
from __future__ import annotations

from typing import Any

from src.live_operations.config import NOTIFICATION_LEVELS


class NotificationEngine:
    def evaluate(self, *, issues: dict[str, Any], anomalies: dict[str, Any]) -> dict[str, Any]:
        alerts: list[dict[str, Any]] = []

        for item in issues.get("open_issues") or []:
            level = self._normalize_level(str(item.get("severity") or "NOTICE"))
            alerts.append(
                {
                    "level": level,
                    "message": item.get("message"),
                    "source": item.get("source"),
                    "check": item.get("check"),
                }
            )

        if not alerts and not anomalies.get("anomalies"):
            alerts.append(
                {
                    "level": "INFO",
                    "message": "Portfolio OS live operations nominal",
                    "source": "rc2",
                    "check": "daily_status",
                }
            )

        by_level = {lvl: [a for a in alerts if a["level"] == lvl] for lvl in NOTIFICATION_LEVELS}
        return {
            "operational_alerts": alerts,
            "alerts_by_level": by_level,
            "alert_count": len(alerts),
            "has_critical": any(a["level"] == "CRITICAL" for a in alerts),
        }

    def _normalize_level(self, raw: str) -> str:
        upper = raw.upper()
        if upper == "ALERT":
            return "CRITICAL"
        if upper in NOTIFICATION_LEVELS:
            return upper
        return "NOTICE"
