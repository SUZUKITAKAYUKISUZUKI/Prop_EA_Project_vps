"""Portfolio governance rules for APM v1."""
from __future__ import annotations

from typing import Any

from src.apm.executive_context import ExecutiveContext


class GovernanceEngine:
    def evaluate(self, context: ExecutiveContext, policies: dict[str, Any]) -> dict[str, Any]:
        flags: list[str] = []
        if policies.get("recovery_mode"):
            flags.append("RECOVERY_GOVERNANCE_ACTIVE")
        if float(context.health.get("health_score") or 0) < 60:
            flags.append("HEALTH_WATCH")
        if context.risk.get("highest_risk_strategy"):
            flags.append(f"RISK_FOCUS_{context.risk['highest_risk_strategy']}")

        return {
            "governance_flags": flags,
            "approval_required": True,
            "auto_execute_allowed": False,
            "governance_level": "STRICT" if flags else "NORMAL",
        }
