"""Governance policy evaluation for APM v1."""
from __future__ import annotations

from src.apm.executive_context import ExecutiveContext


class PolicyEngine:
    def evaluate(self, context: ExecutiveContext) -> dict[str, object]:
        health = float(context.health.get("health_score") or 0)
        recovery_events = int(context.portfolio_state.get("recovery_events") or 0)
        return {
            "allow_profile_switch": health >= 70 and recovery_events == 0,
            "allow_promotion": health >= 60,
            "allow_retirement": True,
            "require_approval_above_confidence": 85.0,
            "max_allocation_shift_pct": 15.0,
            "recovery_mode": str(context.portfolio_state.get("current_state") or "").lower() == "recovery",
        }
