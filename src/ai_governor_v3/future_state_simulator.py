"""Future state trajectory simulation for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG
from src.ai_governor_v3.transition_probability_engine import TransitionProbabilityEngine


class FutureStateSimulator:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        transitions: TransitionProbabilityEngine | None = None,
    ) -> None:
        self._config = config or DEFAULT_CONFIG
        self._transitions = transitions or TransitionProbabilityEngine(owns_connection=False)

    def simulate_future_state(
        self,
        context: GovernorContext,
        *,
        forecasts: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        forecasts = forecasts or {}
        state = context.current_state.lower()
        timeline: list[dict[str, Any]] = [{"day": 0, "state": state.upper()}]

        profile_fc = (forecasts.get("profile") or {}).get("state_forecast") or {}
        for days in self._config.forecast_days:
            projected = profile_fc.get(f"{days}d")
            if projected:
                timeline.append({"day": days, "state": str(projected).upper()})
            else:
                timeline.append({"day": days, "state": self._simulate_at_day(state, days).upper()})

        # Deduplicate consecutive same states except endpoints
        return timeline

    def _simulate_at_day(self, start: str, days: int) -> str:
        state = start
        for day in range(1, days + 1):
            if day % 30 == 0:
                nxt, prob = self._transitions.most_likely_next(state)
                if prob < 0.75:
                    state = nxt
        return state
