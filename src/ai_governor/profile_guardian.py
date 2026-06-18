"""Profile switch guardian for AGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.decision_rules import DEFAULT_RULES, GovernorDecisionRules
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.signals import GuardianSignal


@dataclass
class ProfileAssessment:
    current_profile: str
    recommended_profile: str | None
    profile_drift: bool
    profile_mismatch: bool
    recommended_action: str
    signals: list[GuardianSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_profile": self.current_profile,
            "recommended_profile": self.recommended_profile,
            "profile_drift": self.profile_drift,
            "profile_mismatch": self.profile_mismatch,
            "recommended_action": self.recommended_action,
            "signals": [s.to_dict() for s in self.signals],
        }


class ProfileGuardian:
    def __init__(self, rules: GovernorDecisionRules | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def assess(self, context: GovernorContext) -> ProfileAssessment:
        signals = self.evaluate(context)
        recommended = None
        for sig in signals:
            recommended = sig.reason.get("recommended_profile")
            if recommended:
                break
        mismatch = _profile_mismatch(context)
        drift = bool(context.paae.get("drift_alerts"))
        action = signals[0].decision_type if signals else DecisionType.NO_ACTION.value
        return ProfileAssessment(
            current_profile=context.current_profile,
            recommended_profile=recommended,
            profile_drift=drift,
            profile_mismatch=mismatch,
            recommended_action=action,
            signals=signals,
        )

    def evaluate(self, context: GovernorContext) -> list[GuardianSignal]:
        signals: list[GuardianSignal] = []

        if _profile_mismatch(context):
            recommended = _resolve_profile(context)
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.PROFILE_SWITCH.value,
                    decision=f"Profile mismatch — switch to {recommended}",
                    confidence=88.0,
                    priority="HIGH",
                    source="profile_guardian",
                    expected_benefit=18.0,
                    expected_risk=4.0,
                    reason={
                        "trigger": "profile_mismatch",
                        "current_profile": context.current_profile,
                        "current_state": context.current_state,
                        "recommended_profile": recommended,
                        "recommended_action": DecisionType.PROFILE_SWITCH.value,
                    },
                )
            )
            return signals

        recommended = _resolve_profile(context)
        if recommended and recommended != context.current_profile:
            signals.append(
                GuardianSignal(
                    decision_type=DecisionType.PROFILE_SWITCH.value,
                    decision=f"Switch profile to {recommended}",
                    confidence=80.0,
                    priority="HIGH",
                    source="profile_guardian",
                    expected_benefit=12.0,
                    expected_risk=5.0,
                    reason={
                        "trigger": "profile_state_resolution",
                        "current_profile": context.current_profile,
                        "recommended_profile": recommended,
                        "current_state": context.current_state,
                    },
                )
            )
        return signals


def _profile_mismatch(context: GovernorContext) -> bool:
    state = context.current_state.lower()
    profile_id = context.current_profile.lower()
    profile_type = str(context.profile.get("profile_type") or "").lower()
    challenge_profile = "challenge" in profile_id or profile_type == "challenge"
    return challenge_profile and state in {"funded", "live"}


def _resolve_profile(context: GovernorContext) -> str:
    try:
        from src.services.profile_service import ProfileService

        svc = ProfileService()
        try:
            return svc.resolve_profile_from_state(context.current_state)
        finally:
            svc.close()
    except Exception:
        return context.current_profile
