"""Live operations report builder for RC2."""
from __future__ import annotations

from typing import Any


class LiveReport:
    def build(
        self,
        *,
        profile_id: str,
        briefing: dict[str, Any],
        digest: dict[str, Any],
        metrics: dict[str, Any],
        readiness: dict[str, Any],
        watchdog: dict[str, Any],
        anomalies: dict[str, Any],
        issues: dict[str, Any],
        notifications: dict[str, Any],
        morning: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            "daily_briefing": briefing,
            "daily_digest": digest.get("daily_digest"),
            "digest_summary": digest.get("digest_summary"),
            "morning_report": morning.get("morning_report"),
            **metrics,
            **readiness,
            "watchdog": watchdog,
            "anomalies": anomalies,
            "issues": issues,
            "operational_alerts": notifications.get("operational_alerts"),
            "alerts_by_level": notifications.get("alerts_by_level"),
            "open_alerts": [a for a in notifications.get("operational_alerts") or [] if a.get("level") != "INFO"],
            "definition_of_done": {
                "ai_cio_operational": bool(briefing.get("cio_opinion")),
                "orl_operational": float(briefing.get("readiness") or 0) >= 85,
                "rc1_passed": float(metrics.get("operational_score") or 0) >= 85,
                "rc2_passed": readiness.get("rc2_passed"),
                "operational_score_met": float(metrics.get("operational_score") or 0) >= 90,
                "readiness_met": float(readiness.get("live_readiness") or 0) >= 90,
                "user_action_load_met": int(metrics.get("user_action_load") or 0) <= 3,
                "no_critical_alerts": not issues.get("has_critical"),
                "portfolio_os_complete": readiness.get("portfolio_os_complete"),
            },
            "summary": self._summary(briefing, readiness, metrics, issues),
        }

    def _summary(
        self,
        briefing: dict[str, Any],
        readiness: dict[str, Any],
        metrics: dict[str, Any],
        issues: dict[str, Any],
    ) -> str:
        if readiness.get("portfolio_os_complete"):
            return (
                f"Portfolio OS complete for {briefing.get('date')}: "
                f"operational {metrics.get('operational_score')}, readiness {readiness.get('live_readiness')}."
            )
        return (
            f"Live operations {readiness.get('live_readiness_status')}: "
            f"score {metrics.get('operational_score')}, "
            f"{issues.get('open_issue_count', 0)} open issue(s)."
        )
