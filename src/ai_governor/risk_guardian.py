"""Risk protection guardian for AGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.signals import GuardianSignal


@dataclass
class RiskAssessment:
    current_dd: float
    max_dd: float
    risk_budget_used_pct: float
    risk_budget_remaining_pct: float
    portfolio_risk_score: float
    recovery_probability: float
    recommended_action: str
    signals: list[GuardianSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_dd": self.current_dd,
            "max_dd": self.max_dd,
            "risk_budget_used_pct": self.risk_budget_used_pct,
            "risk_budget_remaining_pct": self.risk_budget_remaining_pct,
            "portfolio_risk_score": self.portfolio_risk_score,
            "recovery_probability": self.recovery_probability,
            "recommended_action": self.recommended_action,
            "signals": [s.to_dict() for s in self.signals],
        }


class RiskGuardian:
    def __init__(self, rules: GovernorDecisionRules | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def assess(self, context: GovernorContext) -> RiskAssessment:
        signals = self.evaluate(context)
        dd_used = _dd_budget_used_pct(context)
        max_dd = float(context.profile.get("total_dd_limit") or 100.0)
        action = DecisionType.NO_ACTION.value
        if signals:
            action = signals[0].decision_type
        recovery_prob = max(0.0, min(100.0, 100.0 - dd_used - context.risk_score * 0.3))
        return RiskAssessment(
            current_dd=context.dd_pct,
            max_dd=max_dd,
            risk_budget_used_pct=dd_used,
            risk_budget_remaining_pct=max(0.0, 100.0 - dd_used),
            portfolio_risk_score=context.risk_score,
            recovery_probability=round(recovery_prob, 1),
            recommended_action=action,
            signals=signals,
        )

    def evaluate(self, context: GovernorContext) -> list[GuardianSignal]:
        rules = self._rules
        signals: list[GuardianSignal] = []
        dd_used = _dd_budget_used_pct(context)

        if dd_used >= rules.dd_force_recovery_pct:
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.ENTER_RECOVERY.value,
                    decision=f"Force Recovery — DD at {dd_used:.1f}% of budget",
                    confidence=98.0,
                    priority="CRITICAL",
                    source="risk_guardian",
                    expected_benefit=15.0,
                    expected_risk=5.0,
                    reason={
                        "trigger": "dd_budget_exceeded",
                        "current_dd": context.dd_pct,
                        "dd_budget_used_pct": dd_used,
                        "threshold": rules.dd_force_recovery_pct,
                        "recommended_action": DecisionType.ENTER_RECOVERY.value,
                    },
                )
            )
        elif dd_used >= rules.dd_recommend_recovery_pct:
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.ENTER_RECOVERY.value,
                    decision=f"Recommend Recovery — DD at {dd_used:.1f}% of budget",
                    confidence=85.0,
                    priority="HIGH",
                    source="risk_guardian",
                    expected_benefit=12.0,
                    expected_risk=8.0,
                    reason={
                        "trigger": "dd_rising",
                        "current_dd": context.dd_pct,
                        "dd_budget_used_pct": dd_used,
                        "threshold": rules.dd_recommend_recovery_pct,
                        "recommended_action": DecisionType.ENTER_RECOVERY.value,
                    },
                )
            )
        elif dd_used >= 55.0:
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.REDUCE_RISK.value,
                    decision=f"Reduce risk exposure — DD at {dd_used:.1f}% of budget",
                    confidence=80.0,
                    priority="MEDIUM",
                    source="risk_guardian",
                    expected_benefit=8.0,
                    expected_risk=6.0,
                    reason={
                        "trigger": "dd_elevated",
                        "current_dd": context.dd_pct,
                        "recommended_action": DecisionType.REDUCE_RISK.value,
                    },
                )
            )

        if context.risk_score >= rules.risk_score_alert_threshold:
            top = (context.prae_v2.get("strategy_risk") or [{}])[0]
            strategy = top.get("strategy")
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.RISK_ALERT.value,
                    decision=f"Elevated portfolio risk score {context.risk_score:.1f}",
                    confidence=82.0,
                    priority="HIGH",
                    source="risk_guardian",
                    strategy_id=str(strategy) if strategy else None,
                    expected_benefit=5.0,
                    expected_risk=20.0,
                    reason={
                        "trigger": "risk_score_elevated",
                        "risk_score": context.risk_score,
                        "threshold": rules.risk_score_alert_threshold,
                        "top_risk_strategy": strategy,
                    },
                )
            )

        if context.health_score < rules.health_at_risk:
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.HEALTH_ALERT.value,
                    decision=f"Portfolio health degraded ({context.health_score:.1f})",
                    confidence=78.0,
                    priority="HIGH",
                    source="risk_guardian",
                    expected_benefit=10.0,
                    expected_risk=15.0,
                    reason={
                        "trigger": "health_degraded",
                        "health_score": context.health_score,
                        "recovery_events": context.recovery_events,
                    },
                )
            )

        if not signals:
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.NO_ACTION.value,
                    decision="Risk within acceptable bounds",
                    confidence=90.0,
                    priority="INFO",
                    source="risk_guardian",
                    reason={"trigger": "risk_stable", "dd_budget_used_pct": dd_used},
                )
            )
        return signals


def _dd_budget_used_pct(context: GovernorContext) -> float:
    if context.dd_pct > 0:
        budget = float(context.profile.get("total_dd_limit") or 100.0)
        if budget > 0 and context.dd_pct <= 100:
            if context.dd_pct <= budget:
                return min(100.0, context.dd_pct / budget * 100.0)
            return context.dd_pct
        return context.dd_pct
    dd_attr = context.prae_v2.get("dd_attribution") or {}
    return float(dd_attr.get("dd_budget_used_pct") or dd_attr.get("portfolio_dd_pct") or 0.0)
