"""Roadmap learning from historical executive outcomes."""
from __future__ import annotations

from typing import Any


class RoadmapLearningEngine:
    def evaluate(
        self,
        *,
        apm_v1_report: dict[str, Any],
        outcomes: list[dict[str, Any]],
        learning: dict[str, Any],
    ) -> dict[str, Any]:
        roadmap = list(apm_v1_report.get("roadmap") or [])
        learned_roadmap: list[dict[str, Any]] = []
        failure_types = {str(o.get("decision_type")) for o in outcomes if o.get("outcome_class") == "FAILURE"}

        for item in roadmap:
            action_type = str(item.get("action_type") or "")
            adjusted = dict(item)
            if action_type in failure_types:
                adjusted["status"] = "REVIEW_REQUIRED"
                adjusted["learning_note"] = "Historical outcomes suggest caution for this action type."
            else:
                adjusted["learning_note"] = "Aligned with positive executive memory."
            learned_roadmap.append(adjusted)

        return {
            "learned_roadmap": learned_roadmap,
            "roadmap_adjustments": sum(1 for i in learned_roadmap if i.get("status") == "REVIEW_REQUIRED"),
            "learning_context": learning.get("improvement_opportunities") or [],
        }
