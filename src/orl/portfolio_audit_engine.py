"""Portfolio data audit for ORL v1."""
from __future__ import annotations

from typing import Any


class PortfolioAuditEngine:
    def evaluate(
        self,
        *,
        profile_id: str,
        ai_cio_report: dict[str, Any] | None,
        cil_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        issues: list[str] = []

        if not profile_id:
            issues.append("Profile ID missing")

        for label, report in (("AI CIO", ai_cio_report), ("CIL", cil_report)):
            if not report:
                issues.append(f"{label} report missing")
                continue
            score = report.get("cio_score") if label == "AI CIO" else report.get("executive_score")
            if score is not None and not (0 <= float(score) <= 100):
                issues.append(f"{label} score out of range: {score}")
            if label == "AI CIO" and not report.get("cio_opinion"):
                issues.append("AI CIO opinion missing")
            if label == "AI CIO" and not report.get("recommendations"):
                issues.append("AI CIO recommendations missing")

        score = max(0.0, 100.0 - len(issues) * 20)
        return {
            "audit_score": round(score, 2),
            "issues": issues,
            "healthy": not issues,
        }
