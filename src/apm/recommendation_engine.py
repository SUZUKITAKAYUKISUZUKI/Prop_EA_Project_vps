"""Executive recommendations for APM v1."""
from __future__ import annotations

from typing import Any

from src.apm.executive_context import ExecutiveContext


class RecommendationEngine:
    def evaluate(
        self,
        context: ExecutiveContext,
        *,
        executive: dict[str, Any],
        actions: list[dict[str, Any]],
        mie_report: dict[str, Any],
    ) -> dict[str, Any]:
        primary = next((a for a in actions if a.get("action_type") != "NO_ACTION"), actions[0] if actions else None)
        strongest = (mie_report.get("strongest_module") or {}).get("module") or "AGE"
        return {
            "recommended_action": primary.get("action_type") if primary else "NO_ACTION",
            "recommended_strategy": primary.get("strategy") if primary else None,
            "primary_trusted_module": strongest,
            "executive_score": executive.get("executive_score"),
            "governance_note": (
                f"Execute via human approval — primary signal from {strongest}, "
                f"executive score {executive.get('executive_score')}."
            ),
            "pending_actions": [a for a in actions if a.get("status") == "PENDING_APPROVAL"],
        }
