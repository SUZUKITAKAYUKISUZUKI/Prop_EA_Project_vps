"""Executive report builder for APM v1."""
from __future__ import annotations

from typing import Any

from src.apm.executive_context import ExecutiveContext


class ExecutiveReport:
    def build(
        self,
        *,
        context: ExecutiveContext,
        executive: dict[str, Any],
        governance: dict[str, Any],
        policies: dict[str, Any],
        recommendations: dict[str, Any],
        opportunities: list[dict[str, Any]],
        risks: list[dict[str, Any]],
        roadmap: list[dict[str, Any]],
        execution_queue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pending = [a for a in execution_queue if a.get("status") == "PENDING_APPROVAL"]
        return {
            "profile_id": context.profile_id,
            "executive_score": executive.get("executive_score"),
            "executive_category": executive.get("executive_category"),
            "executive_health": executive.get("executive_health"),
            "executive_components": executive.get("components"),
            "governance": governance,
            "policies": policies,
            "recommendations": recommendations,
            "recommended_actions": recommendations.get("pending_actions") or pending,
            "opportunities": opportunities,
            "risk_alerts": risks,
            "roadmap": roadmap,
            "execution_queue": execution_queue,
            "approval_queue": pending,
            "context": context.to_dict(),
        }
