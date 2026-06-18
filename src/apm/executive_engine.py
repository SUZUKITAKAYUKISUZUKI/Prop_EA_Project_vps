"""Executive score computation for APM v1."""
from __future__ import annotations

from typing import Any

from src.apm.config import EXECUTIVE_WEIGHTS, executive_category
from src.apm.executive_context import ExecutiveContext


class ExecutiveEngine:
    def evaluate(self, context: ExecutiveContext, mie_report: dict[str, Any]) -> dict[str, Any]:
        governor_confidence = float(
            context.governor_decisions.get("strategic_confidence")
            or context.confidence.get("portfolio_confidence")
            or 70
        )
        portfolio_health = float(context.health.get("health_score") or 70)
        risk_budget = self._risk_budget_score(context)
        trust_score = self._aggregate_trust(mie_report)
        consensus = float(context.confidence.get("consensus_score") or 70)

        executive_score = round(
            EXECUTIVE_WEIGHTS["governor_confidence"] * governor_confidence
            + EXECUTIVE_WEIGHTS["portfolio_health"] * portfolio_health
            + EXECUTIVE_WEIGHTS["risk_budget"] * risk_budget
            + EXECUTIVE_WEIGHTS["trust_score"] * trust_score
            + EXECUTIVE_WEIGHTS["consensus"] * consensus,
            2,
        )

        return {
            "executive_score": executive_score,
            "executive_category": executive_category(executive_score),
            "components": {
                "governor_confidence": round(governor_confidence, 2),
                "portfolio_health": round(portfolio_health, 2),
                "risk_budget": round(risk_budget, 2),
                "trust_score": round(trust_score, 2),
                "consensus": round(consensus, 2),
            },
            "executive_health": portfolio_health,
        }

    def _risk_budget_score(self, context: ExecutiveContext) -> float:
        risks = context.risk.get("strategy_risk") or []
        if not risks:
            return 75.0
        avg_risk = sum(float(r.get("risk_score") or 50) for r in risks) / len(risks)
        return max(0.0, min(100.0, 100.0 - avg_risk * 0.5))

    def _aggregate_trust(self, mie_report: dict[str, Any]) -> float:
        scores = mie_report.get("module_trust_scores") or {}
        if not scores:
            return 70.0
        values = [float(v.get("trust_score") or 0) for v in scores.values() if isinstance(v, dict)]
        return sum(values) / len(values) if values else 70.0
