"""Executive operational summary for ORL v1."""
from __future__ import annotations

from typing import Any


class ExecutiveSummaryEngine:
    def evaluate(
        self,
        *,
        readiness: dict[str, Any],
        system_health: dict[str, Any],
        consistency: dict[str, Any],
        audit: dict[str, Any],
        all_issues: list[str],
    ) -> dict[str, Any]:
        critical = [i for i in all_issues if "missing" in i.lower() or "failed" in i.lower() or "conflict" in i.lower()]
        open_issues = list(dict.fromkeys(all_issues))[:20]

        return {
            "open_operational_issues": open_issues,
            "critical_issue_count": len(critical),
            "critical_issues": critical,
            "can_operate_with_ai_cio_alone": bool(
                readiness.get("production_ready")
                and readiness.get("readiness_score", 0) >= 85
                and not critical
                and consistency.get("healthy")
                and audit.get("healthy")
            ),
            "summary": self._summary(readiness, open_issues),
        }

    def _summary(self, readiness: dict[str, Any], issues: list[str]) -> str:
        score = readiness.get("readiness_score", 0)
        status = readiness.get("readiness_status", "NOT_READY")
        if not issues:
            return f"Portfolio OS operational readiness {status} (score {score}). Safe to operate via AI CIO."
        return f"Portfolio OS readiness {status} (score {score}). {len(issues)} open issue(s) require attention."
