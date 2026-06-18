"""Profile transition forecast for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG
from src.ai_governor_v3.transition_probability_engine import TransitionProbabilityEngine


class ProfileForecaster:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        transitions: TransitionProbabilityEngine | None = None,
    ) -> None:
        self._config = config or DEFAULT_CONFIG
        self._transitions = transitions or TransitionProbabilityEngine(owns_connection=False)

    def forecast(self, context: GovernorContext) -> dict[str, Any]:
        state = context.current_state.lower()
        projections: dict[str, str] = {"current": state.upper()}
        risks: list[str] = []

        for days in self._config.forecast_days:
            projected = self._project_state(state, days)
            projections[f"{days}d"] = projected.upper()
            if state == "funded" and projected == "recovery":
                risks.append("PROFILE_TRANSITION_RISK")

        recommended_profile = None
        try:
            from src.services.profile_service import ProfileService

            svc = ProfileService()
            try:
                recommended_profile = svc.resolve_profile_from_state(projections.get("90d", state).lower())
            finally:
                svc.close()
        except Exception:
            recommended_profile = context.current_profile

        return {
            "current_state": state.upper(),
            "current_profile": context.current_profile,
            "state_forecast": projections,
            "recommended_profile_90d": recommended_profile,
            "profile_transition_risk": risks,
        }

    def _project_state(self, current: str, days: int) -> str:
        steps = max(1, days // 30)
        state = current
        for _ in range(steps):
            nxt, _ = self._transitions.most_likely_next(state)
            if nxt != state:
                recovery_prob = self._transitions.probability(state, "recovery")
                if recovery_prob > 0.15 and state in {"funded", "live"}:
                    state = "recovery"
                else:
                    state = nxt
            else:
                break
        return state
