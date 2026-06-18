"""Decision effectiveness tracking by category."""
from __future__ import annotations

from typing import Any

from src.apm_v2.config import EFFECTIVENESS_CATEGORIES


class DecisionEffectivenessEngine:
    def evaluate(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        buckets: dict[str, list[float]] = {cat: [] for cat in EFFECTIVENESS_CATEGORIES}
        for outcome in outcomes:
            category = self._map_category(str(outcome.get("decision_type") or ""))
            buckets[category].append(float(outcome.get("success_score") or 0))

        effectiveness = {
            self._label(cat): round(sum(scores) / len(scores), 2) if scores else 0.0
            for cat, scores in buckets.items()
        }
        active = [v for v in effectiveness.values() if v > 0]
        decision_accuracy = round(sum(active) / len(active), 2) if active else 0.0

        return {
            "decision_accuracy": decision_accuracy,
            "effectiveness_by_category": effectiveness,
            "evaluated_count": len(outcomes),
            "success_count": sum(1 for o in outcomes if o.get("outcome_class") == "SUCCESS"),
            "failure_count": sum(1 for o in outcomes if o.get("outcome_class") == "FAILURE"),
        }

    def _map_category(self, decision_type: str) -> str:
        upper = decision_type.upper()
        if "PROFILE" in upper:
            return "PROFILE_SWITCH"
        if "ALLOCATION" in upper or "REBALANCE" in upper:
            return "ALLOCATION"
        if "PROMOTE" in upper:
            return "PROMOTION"
        if "RECOVERY" in upper:
            return "RECOVERY"
        if "RETIRE" in upper or "DEMOTE" in upper:
            return "RETIREMENT"
        return "ALLOCATION"

    def _label(self, category: str) -> str:
        return {
            "PROFILE_SWITCH": "profile_switch_accuracy",
            "ALLOCATION": "allocation_accuracy",
            "PROMOTION": "promotion_accuracy",
            "RECOVERY": "recovery_accuracy",
            "RETIREMENT": "retirement_accuracy",
        }.get(category, f"{category.lower()}_accuracy")
