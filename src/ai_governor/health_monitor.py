"""Portfolio health synthesis for AGE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.risk_guardian import _dd_budget_used_pct


@dataclass
class PortfolioHealthSnapshot:
    health_score: float
    health_status: str
    risk_level: str
    state: str
    profile: str
    risk_score: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "health_score": self.health_score,
            "health_status": self.health_status,
            "risk_level": self.risk_level,
            "state": self.state,
            "profile": self.profile,
            "risk_score": self.risk_score,
            "components": self.components,
        }


class HealthMonitor:
    """
    Governor Health Score:
    0.25 Portfolio Health + 0.25 Risk Budget Remaining + 0.20 State Stability
    + 0.15 Recovery Frequency + 0.15 Strategy Health
    """

    def __init__(self, rules: GovernorDecisionRules | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def evaluate(self, context: GovernorContext) -> PortfolioHealthSnapshot:
        portfolio_health = float(context.health_score)
        dd_used = _dd_budget_used_pct(context)
        risk_budget_remaining = max(0.0, 100.0 - dd_used)
        state_stability = float(
            context.state_analytics.get("funded_stability_score")
            or context.state_analytics.get("health_score")
            or portfolio_health
        )
        recovery_frequency = max(0.0, 100.0 - min(100.0, context.recovery_events * 20.0))
        strategy_scores = list(context.strategy_scores.values())
        strategy_health = sum(strategy_scores) / len(strategy_scores) if strategy_scores else portfolio_health

        composite = round(
            portfolio_health * 0.25
            + risk_budget_remaining * 0.25
            + state_stability * 0.20
            + recovery_frequency * 0.15
            + strategy_health * 0.15,
            1,
        )
        composite = max(0.0, min(100.0, composite))
        return PortfolioHealthSnapshot(
            health_score=composite,
            health_status=self._status(composite),
            risk_level=context.risk_level,
            state=context.current_state,
            profile=context.current_profile,
            risk_score=context.risk_score,
            components={
                "portfolio_health": round(portfolio_health, 1),
                "risk_budget_remaining": round(risk_budget_remaining, 1),
                "state_stability": round(state_stability, 1),
                "recovery_frequency": round(recovery_frequency, 1),
                "strategy_health": round(strategy_health, 1),
            },
        )

    def _status(self, score: float) -> str:
        rules = self._rules
        if score >= rules.health_excellent:
            return "EXCELLENT"
        if score >= rules.health_good:
            return "GOOD"
        if score >= rules.health_fair:
            return "FAIR"
        if score >= rules.health_at_risk:
            return "WARNING"
        return "CRITICAL"
