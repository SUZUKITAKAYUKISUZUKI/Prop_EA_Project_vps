"""Executive summary and investment score for CIL v1."""
from __future__ import annotations

from typing import Any

from src.cio_intelligence.config import EXECUTIVE_INVESTMENT_WEIGHTS


class ExecutiveSummaryEngine:
    def evaluate(
        self,
        *,
        investment_states: list[str],
        opportunity: dict[str, Any],
        risk: dict[str, Any],
        confidence: dict[str, Any],
        capital_efficiency: dict[str, Any],
        bundle: dict[str, Any],
    ) -> dict[str, Any]:
        portfolio_health = float(
            ((bundle.get("prae_v2") or {}).get("health_report") or {}).get("health_score") or 70
        )
        intelligence_trust = self._intelligence_trust(bundle)

        executive_score = round(
            EXECUTIVE_INVESTMENT_WEIGHTS["portfolio_health"] * portfolio_health
            + EXECUTIVE_INVESTMENT_WEIGHTS["capital_efficiency"] * float(capital_efficiency.get("capital_efficiency") or 0)
            + EXECUTIVE_INVESTMENT_WEIGHTS["opportunity_score"] * float(opportunity.get("opportunity_score") or 0)
            + EXECUTIVE_INVESTMENT_WEIGHTS["risk_score"] * float(risk.get("risk_score") or 0)
            + EXECUTIVE_INVESTMENT_WEIGHTS["confidence_score"] * float(confidence.get("confidence_score") or 0)
            + EXECUTIVE_INVESTMENT_WEIGHTS["intelligence_trust"] * intelligence_trust,
            2,
        )

        return {
            "investment_state": investment_states,
            "executive_score": executive_score,
            "opportunity_score": opportunity.get("opportunity_score"),
            "risk_score": risk.get("risk_score"),
            "confidence_score": confidence.get("confidence_score"),
            "capital_efficiency": capital_efficiency.get("capital_efficiency"),
            "intelligence_trust": round(intelligence_trust, 2),
            "portfolio_health": round(portfolio_health, 2),
            "top_opportunity": opportunity.get("top_opportunity"),
            "top_risk": risk.get("top_risk"),
            "executive_components": {
                "portfolio_health": round(portfolio_health, 2),
                "capital_efficiency": capital_efficiency.get("capital_efficiency"),
                "opportunity_score": opportunity.get("opportunity_score"),
                "risk_score": risk.get("risk_score"),
                "confidence_score": confidence.get("confidence_score"),
                "intelligence_trust": round(intelligence_trust, 2),
            },
        }

    def _intelligence_trust(self, bundle: dict[str, Any]) -> float:
        mie = bundle.get("mie") or {}
        scores = mie.get("module_trust_scores") or {}
        if not scores:
            return 70.0
        values = [float(v.get("trust_score") or 0) for v in scores.values() if isinstance(v, dict)]
        return sum(values) / len(values) if values else 70.0
