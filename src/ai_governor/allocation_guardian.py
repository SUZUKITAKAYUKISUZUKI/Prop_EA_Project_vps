"""Allocation rebalance guardian for AGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.signals import GuardianSignal


@dataclass
class AllocationAssessment:
    allocation_drift: bool
    rebalance_recommended: bool
    score_improvement: float
    recommended_action: str
    signals: list[GuardianSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_drift": self.allocation_drift,
            "rebalance_recommended": self.rebalance_recommended,
            "score_improvement": self.score_improvement,
            "recommended_action": self.recommended_action,
            "signals": [s.to_dict() for s in self.signals],
        }


class AllocationGuardian:
    def __init__(self, rules: GovernorDecisionRules | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def assess(self, context: GovernorContext) -> AllocationAssessment:
        signals = self.evaluate(context)
        gap = _score_gap(context)
        drift = bool(context.paae.get("drift_alerts"))
        rebalance = any(s.decision_type == DecisionType.ALLOCATION_REBALANCE.value for s in signals)
        action = DecisionType.ALLOCATION_REBALANCE.value if rebalance else DecisionType.NO_ACTION.value
        return AllocationAssessment(
            allocation_drift=drift,
            rebalance_recommended=rebalance,
            score_improvement=gap,
            recommended_action=action,
            signals=signals,
        )

    def evaluate(self, context: GovernorContext) -> list[GuardianSignal]:
        rules = self._rules
        signals: list[GuardianSignal] = []
        gap = _score_gap(context)

        if gap >= rules.rebalance_score_gap_pct:
            paae_agrees = bool(context.paae.get("recommended_weights"))
            pdts_agrees = gap >= rules.rebalance_score_gap_pct
            confidence = 95.0 if paae_agrees and pdts_agrees else min(95.0, 70.0 + gap)
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.ALLOCATION_REBALANCE.value,
                    decision=f"Recommend rebalance — PDTS score improvement {gap:.1f} pts",
                    confidence=confidence,
                    priority="HIGH" if gap >= 15.0 else "MEDIUM",
                    source="allocation_guardian",
                    expected_benefit=gap,
                    expected_risk=max(2.0, gap * 0.2),
                    reason={
                        "trigger": "allocation_improvement",
                        "current_score": _scenario_score(context.pdts, "current"),
                        "recommended_score": _scenario_score(context.pdts, "recommended"),
                        "improvement": gap,
                        "portfolio_fit_gain": _portfolio_fit_gain(context),
                        "recommended_action": DecisionType.ALLOCATION_REBALANCE.value,
                    },
                )
            )

        drift = context.paae.get("drift_alerts") or []
        if drift:
            top = drift[0]
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.ALLOCATION_REBALANCE.value,
                    decision=f"PAAE allocation drift for {top.get('strategy')}",
                    confidence=75.0,
                    priority="MEDIUM",
                    source="allocation_guardian",
                    strategy_id=str(top.get("strategy") or ""),
                    expected_benefit=5.0,
                    expected_risk=3.0,
                    reason={"trigger": "allocation_drift", "drift_alert": top},
                )
            )

        return signals


def _score_gap(context: GovernorContext) -> float:
    return _scenario_score(context.pdts, "recommended") - _scenario_score(context.pdts, "current")


def _scenario_score(pdts: dict, key: str) -> float:
    cmp = pdts.get("scenario_comparison") or {}
    row = cmp.get(key) or {}
    return float(row.get("score") or row.get("recommendation_score") or 0.0)


def _portfolio_fit_gain(context: GovernorContext) -> float:
    cmp = context.pdts.get("scenario_comparison") or {}
    rec = cmp.get("recommended") or {}
    return float(rec.get("portfolio_fit_gain") or 0.0)
