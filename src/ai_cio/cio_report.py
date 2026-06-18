"""Final AI CIO report builder."""
from __future__ import annotations

from typing import Any


class CioReport:
    def build(
        self,
        *,
        profile_id: str,
        executive: dict[str, Any],
        opinion: dict[str, Any],
        opportunity: dict[str, Any],
        risk: dict[str, Any],
        memory: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            **executive,
            "opinion_rationale": opinion.get("opinion_rationale"),
            "opportunity_report": opportunity,
            "risk_report": risk,
            "cio_memory": {
                "past_successes": memory.get("past_successes"),
                "past_mistakes": memory.get("past_mistakes"),
                "recurring_problems": memory.get("recurring_problems"),
            },
            "executive_lessons": memory.get("executive_lessons"),
            "recommendations": recommendations.get("recommendations"),
            "recommended_actions": recommendations.get("recommended_actions"),
            "top_recommendation": recommendations.get("top_recommendation"),
            "advisor_notice": "Human approval remains mandatory. AI CIO does not execute trades or modify allocations.",
        }
