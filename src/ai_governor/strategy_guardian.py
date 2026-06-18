"""Strategy lifecycle guardian using SLM v3."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.signals import GuardianSignal


@dataclass
class StrategyAssessment:
    promotion_candidates: list[str]
    demotion_candidates: list[str]
    retirement_candidates: list[str]
    recommended_action: str
    signals: list[GuardianSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_candidates": self.promotion_candidates,
            "demotion_candidates": self.demotion_candidates,
            "retirement_candidates": self.retirement_candidates,
            "recommended_action": self.recommended_action,
            "signals": [s.to_dict() for s in self.signals],
        }


class StrategyGuardian:
    def __init__(self, rules: GovernorDecisionRules | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def assess(self, context: GovernorContext) -> StrategyAssessment:
        signals = self.evaluate(context)
        promotions = [s.strategy_id for s in signals if s.decision_type == DecisionType.PROMOTE_STRATEGY.value and s.strategy_id]
        demotions = [s.strategy_id for s in signals if s.decision_type == DecisionType.DEMOTE_STRATEGY.value and s.strategy_id]
        retirements = [s.strategy_id for s in signals if s.decision_type == DecisionType.RETIRE_STRATEGY.value and s.strategy_id]
        action = signals[0].decision_type if signals else DecisionType.NO_ACTION.value
        return StrategyAssessment(
            promotion_candidates=promotions,
            demotion_candidates=demotions,
            retirement_candidates=retirements,
            recommended_action=action,
            signals=signals,
        )

    def evaluate(self, context: GovernorContext) -> list[GuardianSignal]:
        rules = self._rules
        signals: list[GuardianSignal] = []
        slm = context.slm

        for candidate in slm.get("promotion_candidates") or []:
            strategy = str(candidate.get("strategy") or "")
            next_stage = str(candidate.get("next_stage") or "")
            score = float(candidate.get("score") or 0.0)
            fit = float(candidate.get("portfolio_fit_score") or 0.0)
            if not strategy:
                continue
            if next_stage == "CORE":
                if score >= rules.core_promote_score_threshold and fit >= rules.core_promote_fit_threshold:
                    signals.append(
                        _strategy_signal(
                            DecisionType.PROMOTE_STRATEGY.value,
                            f"Promote {strategy} to Core",
                            strategy,
                            score,
                            fit,
                            next_stage,
                            confidence=94.0,
                            priority="HIGH",
                            expected_benefit=fit * 0.2,
                        )
                    )
            elif next_stage == "PRODUCTION":
                if score >= rules.promote_score_threshold and fit >= rules.promote_fit_threshold:
                    signals.append(
                        _strategy_signal(
                            DecisionType.PROMOTE_STRATEGY.value,
                            f"Promote {strategy} to Production",
                            strategy,
                            score,
                            fit,
                            next_stage,
                            confidence=88.0,
                            priority="MEDIUM",
                            expected_benefit=fit * 0.15,
                        )
                    )

        for candidate in slm.get("retirement_candidates") or []:
            strategy = str(candidate.get("strategy") or "")
            score = float(candidate.get("score") or 0.0)
            fit = float(candidate.get("portfolio_fit_score") or 0.0)
            if not strategy:
                continue
            if fit <= rules.retire_fit_threshold or score <= rules.retire_score_threshold:
                signals.append(
                    _strategy_signal(
                        DecisionType.RETIRE_STRATEGY.value,
                        f"Retire {strategy} — fit/score deteriorated",
                        strategy,
                        score,
                        fit,
                        "RETIRED",
                        confidence=86.0,
                        priority="HIGH",
                        expected_risk=20.0,
                    )
                )
            elif fit <= rules.demote_fit_threshold:
                signals.append(
                    _strategy_signal(
                        DecisionType.DEMOTE_STRATEGY.value,
                        f"Demote {strategy} to Recovery",
                        strategy,
                        score,
                        fit,
                        "RECOVERY",
                        confidence=84.0,
                        priority="MEDIUM",
                        expected_risk=10.0,
                    )
                )

        for row in slm.get("lifecycle_decisions") or []:
            if str(row.get("decision")) == "DEMOTE" and row.get("strategy"):
                strategy = str(row["strategy"])
                signals.append(
                    GuardianSignal(
                        decision_type=DecisionType.DEMOTE_STRATEGY.value,
                        decision=f"SLM recommends demotion for {strategy}",
                        confidence=80.0,
                        priority="MEDIUM",
                        source="strategy_guardian",
                        strategy_id=strategy,
                        reason={"slm_decision": row},
                    )
                )

        return signals


def _strategy_signal(
    decision_type: str,
    decision: str,
    strategy: str,
    score: float,
    fit: float,
    next_stage: str,
    *,
    confidence: float,
    priority: str,
    expected_benefit: float = 0.0,
    expected_risk: float = 0.0,
) -> GuardianSignal:
    return GuardianSignal(
        decision_type=decision_type,
        decision=decision,
        confidence=confidence,
        priority=priority,
        source="strategy_guardian",
        strategy_id=strategy,
        expected_benefit=expected_benefit,
        expected_risk=expected_risk,
        reason={
            "strategy": strategy,
            "score": score,
            "portfolio_fit": fit,
            "next_stage": next_stage,
            "core": next_stage == "CORE",
        },
    )
