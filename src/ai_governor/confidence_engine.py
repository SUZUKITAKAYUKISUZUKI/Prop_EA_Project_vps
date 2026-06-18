"""Cross-module agreement confidence scoring for AGE."""
from __future__ import annotations

from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.signals import GuardianSignal


class ConfidenceEngine:
    """Compute 0-100 confidence from guardian agreement and upstream alignment."""

    def __init__(self, rules: GovernorDecisionRules | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def score_signal_group(self, group: list[GuardianSignal], context: GovernorContext) -> float:
        if not group:
            return 0.0
        primary = max(group, key=lambda s: s.confidence)
        agreement = self._agreement(group, context)
        return self.blend(primary.confidence, agreement)

    def score_cross_module(
        self,
        *,
        paae_agrees: bool,
        pdts_agrees: bool,
        prae_agrees: bool,
        slm_agrees: bool,
        base: float,
    ) -> float:
        votes = sum([paae_agrees, pdts_agrees, prae_agrees, slm_agrees])
        if votes >= 4:
            return min(100.0, max(base, self._rules.high_confidence))
        if votes == 3:
            return min(100.0, max(base, self._rules.medium_confidence + 5))
        if votes == 2:
            return max(self._rules.low_confidence, base * 0.85)
        return max(40.0, base * 0.7)

    def blend(self, primary: float, agreement: float) -> float:
        rules = self._rules
        blended = primary * 0.65 + agreement * 0.35
        if blended >= rules.high_confidence - 5:
            return min(100.0, round(blended + 5.0, 1))
        if blended >= rules.medium_confidence:
            return round(blended, 1)
        return round(max(rules.low_confidence - 10, blended), 1)

    def agreement_score(self, group: list[GuardianSignal], context: GovernorContext) -> float:
        return self._agreement(group, context)

    def _agreement(self, group: list[GuardianSignal], context: GovernorContext) -> float:
        if len(group) <= 1:
            return 75.0
        sources = {s.source for s in group}
        base = min(100.0, 60.0 + len(sources) * 10.0)
        conf_avg = sum(s.confidence for s in group) / len(group)
        health_bonus = 5.0 if context.health_score >= 70.0 else -5.0
        return max(0.0, min(100.0, (base + conf_avg) / 2.0 + health_bonus))
