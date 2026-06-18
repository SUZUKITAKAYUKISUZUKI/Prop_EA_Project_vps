"""Morning report wrapper for RC2."""
from __future__ import annotations

from typing import Any


class MorningReportEngine:
    def evaluate(self, *, briefing: dict[str, Any], notifications: dict[str, Any]) -> dict[str, Any]:
        return {
            "morning_report": {
                **briefing,
                "alerts": notifications.get("operational_alerts"),
                "alert_count": notifications.get("alert_count"),
            },
            "generated_at": briefing.get("date"),
        }
