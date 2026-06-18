"""Risk synthesis for AI CIO from CIL report."""
from __future__ import annotations

from typing import Any


class CioRiskEngine:
    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        detail = context.get("risk_detail") or {}
        items = list(context.get("top_risks") or [])

        return {
            "top_risk": context.get("top_risk"),
            "risk_score": context.get("risk_score"),
            "dd_risk": detail.get("dd_risk"),
            "recovery_probability": detail.get("recovery_probability"),
            "risk_budget_consumption": detail.get("risk_budget_consumption"),
            "strategy_concentration": detail.get("strategy_concentration"),
            "portfolio_fragility": detail.get("portfolio_fragility"),
            "risks": items,
        }
