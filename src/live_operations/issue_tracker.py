"""Issue tracking for RC2 live operations."""
from __future__ import annotations

from typing import Any


class IssueTracker:
    def evaluate(
        self,
        *,
        watchdog: dict[str, Any],
        anomalies: dict[str, Any],
        production: dict[str, Any] | None,
    ) -> dict[str, Any]:
        open_issues: list[dict[str, Any]] = []

        for item in watchdog.get("issues") or []:
            open_issues.append({**item, "source": "watchdog"})

        for item in anomalies.get("anomalies") or []:
            open_issues.append(
                {
                    "check": item.get("type"),
                    "severity": item.get("severity", "NOTICE"),
                    "message": item.get("message"),
                    "source": "anomaly",
                    "category": item.get("severity", "NOTICE"),
                }
            )

        for issue in (production or {}).get("open_production_issues") or []:
            if isinstance(issue, str):
                open_issues.append(
                    {"check": "production", "severity": "NOTICE", "message": issue, "source": "rc1", "category": "NOTICE"}
                )

        critical = [i for i in open_issues if str(i.get("severity")).upper() in ("CRITICAL", "ALERT")]
        return {
            "open_issues": open_issues[:30],
            "open_issue_count": len(open_issues),
            "critical_issue_count": len(critical),
            "has_critical": bool(critical),
        }
