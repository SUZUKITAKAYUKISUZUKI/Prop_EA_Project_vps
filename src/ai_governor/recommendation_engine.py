"""Governor recommendation generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.decision_engine import GovernorDecision
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.explainability_engine import ExplainabilityEngine
from src.ai_governor.governor_context import GovernorContext


@dataclass
class GovernorRecommendation:
    action: str
    priority: str
    confidence: float
    reason: str
    expected_benefit: float
    expected_risk: float
    reason_json: dict[str, Any] = field(default_factory=dict)
    status: str = "OPEN"
    strategy_id: str | None = None
    category: str = "MEDIUM"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "priority": self.priority,
            "confidence": self.confidence,
            "reason": self.reason,
            "expected_benefit": self.expected_benefit,
            "expected_risk": self.expected_risk,
            "reason_json": self.reason_json,
            "status": self.status,
            "strategy_id": self.strategy_id,
            "category": self.category,
            "recommendation": self.reason,
            "decision_type": self.action,
        }


class RecommendationEngine:
    def __init__(self, explainability: ExplainabilityEngine | None = None) -> None:
        self._explain = explainability or ExplainabilityEngine()

    def build(
        self,
        context: GovernorContext,
        decisions: list[GovernorDecision],
    ) -> list[GovernorRecommendation]:
        recommendations: list[GovernorRecommendation] = []
        for decision in decisions:
            if decision.decision_type == DecisionType.NO_ACTION.value:
                continue
            category = _category_for(decision)
            benefit, risk = _estimate_impact(decision)
            recommendations.append(
                GovernorRecommendation(
                    action=decision.decision_type,
                    priority=decision.priority,
                    confidence=decision.confidence,
                    reason=decision.decision,
                    expected_benefit=benefit,
                    expected_risk=risk,
                    reason_json=decision.reason_json,
                    strategy_id=decision.strategy_id,
                    category=category,
                )
            )

        recommendations.extend(_opportunity_recommendations(context, self._explain))
        recommendations.sort(key=lambda r: (_priority_rank(r.priority), -r.confidence))
        return recommendations


def _category_for(decision: GovernorDecision) -> str:
    mapping = {
        DecisionType.RISK_ALERT.value: "CRITICAL",
        DecisionType.HEALTH_ALERT.value: "CRITICAL",
        DecisionType.ENTER_RECOVERY.value: "CRITICAL",
        DecisionType.RETIRE_STRATEGY.value: "HIGH",
        DecisionType.DEMOTE_STRATEGY.value: "HIGH",
        DecisionType.PROFILE_SWITCH.value: "HIGH",
        DecisionType.REDUCE_RISK.value: "HIGH",
        DecisionType.ALLOCATION_REBALANCE.value: "MEDIUM",
        DecisionType.PROMOTE_STRATEGY.value: "MEDIUM",
        DecisionType.EXIT_RECOVERY.value: "MEDIUM",
    }
    return mapping.get(decision.decision_type, decision.priority)


def _estimate_impact(decision: GovernorDecision) -> tuple[float, float]:
    reason = decision.reason_json or {}
    benefit = float(reason.get("expected_benefit") or reason.get("improvement") or 5.0)
    risk = float(reason.get("expected_risk") or 5.0)
    if decision.decision_type in {DecisionType.RISK_ALERT.value, DecisionType.HEALTH_ALERT.value}:
        risk = max(risk, 15.0)
    return round(benefit, 1), round(risk, 1)


def _priority_rank(priority: str) -> int:
    order = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
    return order.get(priority.upper(), 0)


def _opportunity_recommendations(
    context: GovernorContext,
    explain: ExplainabilityEngine,
) -> list[GovernorRecommendation]:
    items: list[GovernorRecommendation] = []
    for row in context.slm.get("promotion_candidates") or []:
        strategy = str(row.get("strategy") or "")
        fit = float(row.get("portfolio_fit_score") or 0.0)
        if not strategy or fit < 60.0:
            continue
        reason_json = {
            "trigger": "promotion_opportunity",
            "strategy": strategy,
            "portfolio_fit": fit,
            "next_stage": row.get("next_stage"),
        }
        items.append(
            GovernorRecommendation(
                action=DecisionType.PROMOTE_STRATEGY.value,
                priority="LOW",
                confidence=min(90.0, fit),
                reason=f"Opportunity: promote {strategy} (fit {fit:.1f})",
                expected_benefit=fit * 0.1,
                expected_risk=3.0,
                reason_json=reason_json,
                strategy_id=strategy,
                category="INFO",
            )
        )
    gain = float(
        ((context.pdts.get("scenario_comparison") or {}).get("recommended") or {}).get("portfolio_fit_gain") or 0.0
    )
    if gain >= 5.0:
        items.append(
            GovernorRecommendation(
                action=DecisionType.ALLOCATION_REBALANCE.value,
                priority="LOW",
                confidence=75.0,
                reason=f"PDTS portfolio fit gain available (+{gain:.1f})",
                expected_benefit=gain,
                expected_risk=2.0,
                reason_json={"trigger": "portfolio_fit_gain", "portfolio_fit_gain": gain},
                category="INFO",
            )
        )
    return items
