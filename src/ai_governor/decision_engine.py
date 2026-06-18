"""Governor decision aggregation and confidence scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.confidence_engine import ConfidenceEngine
from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.explainability_engine import ExplainabilityEngine
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.signals import GuardianSignal


@dataclass
class GovernorDecision:
    decision_type: str
    decision: str
    confidence: float
    reason_json: dict[str, Any]
    source_state: str
    profile_id: str
    strategy_id: str | None = None
    priority: str = "MEDIUM"
    sources: list[str] = field(default_factory=list)
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason_json": self.reason_json,
            "profile": self.profile_id,
            "state": self.source_state,
            "source_state": self.source_state,
            "profile_id": self.profile_id,
            "strategy_id": self.strategy_id,
            "priority": self.priority,
            "sources": self.sources,
            "executed": self.executed,
        }


class DecisionEngine:
    def __init__(
        self,
        rules: GovernorDecisionRules | None = None,
        explainability: ExplainabilityEngine | None = None,
        confidence: ConfidenceEngine | None = None,
    ) -> None:
        self._rules = rules or DEFAULT_RULES
        self._explain = explainability or ExplainabilityEngine()
        self._confidence = confidence or ConfidenceEngine(self._rules)

    def evaluate(
        self,
        context: GovernorContext,
        signals: list[GuardianSignal],
    ) -> list[GovernorDecision]:
        actionable = [s for s in signals if s.decision_type != DecisionType.NO_ACTION.value]
        if not actionable:
            return [self._no_action(context, confidence=90.0, reason="All guardians report stable conditions")]

        grouped = _group_signals(actionable)
        decisions: list[GovernorDecision] = []
        ctx_summary = context.to_dict()

        for decision_type, group in grouped.items():
            primary = _pick_primary(group)
            confidence = self._confidence.score_signal_group(group, context)
            sources = sorted({s.source for s in group})
            reason = self._explain.build_reason(primary, context_summary=ctx_summary)
            reason["agreement_score"] = round(
                self._confidence.agreement_score(group, context),
                1,
            )
            reason["supporting_sources"] = sources
            reason["recommended_action"] = decision_type

            decisions.append(
                GovernorDecision(
                    decision_type=decision_type,
                    decision=primary.decision,
                    confidence=confidence,
                    reason_json=reason,
                    source_state=context.source_state,
                    profile_id=context.profile_id,
                    strategy_id=primary.strategy_id,
                    priority=_max_priority(group),
                    sources=sources,
                )
            )

        decisions.sort(key=lambda d: (_priority_rank(d.priority), -d.confidence))
        return decisions

    def _no_action(self, context: GovernorContext, *, confidence: float, reason: str) -> GovernorDecision:
        reason_json = {
            "trigger": "stable_conditions",
            "decision": DecisionType.NO_ACTION.value,
            "decision_text": reason,
            "confidence": confidence,
            "recommended_action": DecisionType.NO_ACTION.value,
            "health_score": context.health_score,
            "portfolio_fit": context.portfolio_fit,
        }
        return GovernorDecision(
            decision_type=DecisionType.NO_ACTION.value,
            decision=reason,
            confidence=confidence,
            reason_json=reason_json,
            source_state=context.source_state,
            profile_id=context.profile_id,
            priority="INFO",
            sources=["decision_engine"],
        )


def _group_signals(signals: list[GuardianSignal]) -> dict[str, list[GuardianSignal]]:
    grouped: dict[str, list[GuardianSignal]] = {}
    for signal in signals:
        grouped.setdefault(signal.decision_type, []).append(signal)
    return grouped


def _pick_primary(group: list[GuardianSignal]) -> GuardianSignal:
    return max(group, key=lambda s: (_priority_rank(s.priority), s.confidence))


def _max_priority(group: list[GuardianSignal]) -> str:
    return max(group, key=lambda s: _priority_rank(s.priority)).priority


def _priority_rank(priority: str) -> int:
    order = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
    return order.get(priority.upper(), 0)
