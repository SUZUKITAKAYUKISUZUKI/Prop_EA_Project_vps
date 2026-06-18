"""Account state guardian for AGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.signals import GuardianSignal


@dataclass
class StateAssessment:
    current_state: str
    transition_ready: bool
    recommended_action: str
    summary: str
    signals: list[GuardianSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state,
            "transition_ready": self.transition_ready,
            "recommended_action": self.recommended_action,
            "summary": self.summary,
            "signals": [s.to_dict() for s in self.signals],
        }


class StateGuardian:
    def __init__(self, rules: GovernorDecisionRules | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def assess(self, context: GovernorContext) -> StateAssessment:
        signals = self.evaluate(context)
        action = signals[0].decision_type if signals else DecisionType.NO_ACTION.value
        summary = signals[0].decision if signals else "State stable"
        transition = action in {
            DecisionType.PROFILE_SWITCH.value,
            DecisionType.EXIT_RECOVERY.value,
            DecisionType.ENTER_RECOVERY.value,
        }
        return StateAssessment(
            current_state=context.current_state,
            transition_ready=transition,
            recommended_action=action,
            summary=summary,
            signals=signals,
        )

    def evaluate(self, context: GovernorContext) -> list[GuardianSignal]:
        signals: list[GuardianSignal] = []
        state = context.current_state.lower()
        analytics = context.state_analytics

        if state == "challenge" and _challenge_passed(analytics, context):
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.PROFILE_SWITCH.value,
                    decision="Challenge Passed — Enter Funded profile",
                    confidence=92.0,
                    priority="HIGH",
                    source="state_guardian",
                    expected_benefit=20.0,
                    expected_risk=5.0,
                    reason={
                        "trigger": "challenge_passed",
                        "current_state": state,
                        "recommended_action": DecisionType.PROFILE_SWITCH.value,
                    },
                )
            )

        if state == "recovery":
            if _recovery_resolved(analytics, context):
                signals.append(
                    GuardianSignal(
                        decision_type=DecisionType.EXIT_RECOVERY.value,
                        decision="Recovery Completed — recommend Funded profile",
                        confidence=88.0,
                        priority="HIGH",
                        source="state_guardian",
                        expected_benefit=15.0,
                        expected_risk=6.0,
                        reason={
                            "trigger": "recovery_completed",
                            "recommended_action": DecisionType.EXIT_RECOVERY.value,
                        },
                    )
                )
            elif _recovery_worsening(analytics, context):
                signals.append(
                    GuardianSignal(
                        decision_type=DecisionType.NO_ACTION.value,
                        decision="Recovery Needed — maintain Recovery profile",
                        confidence=90.0,
                        priority="MEDIUM",
                        source="state_guardian",
                        reason={
                            "trigger": "recovery_worsening",
                            "health_score": context.health_score,
                            "dd_pct": context.dd_pct,
                        },
                    )
                )

        if state == "live" and context.dd_pct >= 60.0:
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.RISK_ALERT.value,
                    decision="Live Account Warning — elevated drawdown",
                    confidence=85.0,
                    priority="HIGH",
                    source="state_guardian",
                    expected_risk=25.0,
                    reason={"trigger": "live_dd_warning", "current_dd": context.dd_pct},
                )
            )

        if state == "funded" and not signals:
            stability = float(analytics.get("funded_stability_score") or 0.0)
            if stability >= 70.0:
                signals.append(
                    GuardianSignal(
                        decision_type=DecisionType.NO_ACTION.value,
                        decision="Funded state stable",
                        confidence=85.0,
                        priority="INFO",
                        source="state_guardian",
                        reason={"funded_stability_score": stability},
                    )
                )

        return signals


def _challenge_passed(analytics: dict[str, Any], context: GovernorContext) -> bool:
    if analytics.get("challenge_passed"):
        return True
    return context.pass_rate >= 95.0 and context.health_score >= 70.0


def _recovery_resolved(analytics: dict[str, Any], context: GovernorContext) -> bool:
    return (
        context.health_score >= 75.0
        and context.dd_pct < 50.0
        and context.recovery_events <= 1
        and context.pass_rate >= 90.0
    )


def _recovery_worsening(analytics: dict[str, Any], context: GovernorContext) -> bool:
    return context.health_score < 60.0 or context.dd_pct >= 70.0 or context.recovery_events >= 2
