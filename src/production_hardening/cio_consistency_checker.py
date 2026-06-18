"""CIO consistency checks for RC1."""
from __future__ import annotations

from typing import Any


class CioConsistencyChecker:
    def evaluate(
        self,
        *,
        ai_cio_report: dict[str, Any] | None,
        orl_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        issues: list[str] = []
        if not ai_cio_report:
            issues.append("AI CIO report missing")
            return {"ai_cio_availability": 0.0, "issues": issues, "healthy": False}

        required = ("cio_score", "cio_opinion", "recommendations", "top_opportunity", "top_risk")
        present = sum(1 for k in required if ai_cio_report.get(k) is not None)
        if not ai_cio_report.get("recommendations"):
            issues.append("AI CIO recommendations not visible")

        if orl_report and not orl_report.get("can_operate_with_ai_cio_alone"):
            issues.append("ORL reports AI CIO alone operation not ready")

        score = round((present / len(required)) * 100, 2)
        if issues:
            score = min(score, max(0.0, score - len(issues) * 10))

        return {
            "ai_cio_availability": score,
            "issues": issues,
            "healthy": score >= 85 and not issues,
            "keys_present": present,
        }
