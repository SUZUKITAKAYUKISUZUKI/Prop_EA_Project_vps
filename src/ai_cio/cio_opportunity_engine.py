"""Opportunity synthesis for AI CIO from CIL report."""
from __future__ import annotations

from typing import Any


class CioOpportunityEngine:
    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        detail = context.get("opportunity_detail") or {}
        items = list(context.get("top_opportunities") or [])

        return {
            "top_opportunity": context.get("top_opportunity"),
            "opportunity_score": context.get("opportunity_score"),
            "expected_r": detail.get("expected_r"),
            "expected_pf": detail.get("expected_pf"),
            "pass_probability": detail.get("pass_probability"),
            "future_health": detail.get("future_health"),
            "growth_potential": detail.get("growth_potential"),
            "opportunities": items,
        }
