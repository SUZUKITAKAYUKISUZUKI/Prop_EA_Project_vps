"""Recovery probability forecast for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.risk_guardian import _dd_budget_used_pct
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG
from src.ai_governor_v3.transition_probability_engine import TransitionProbabilityEngine
from src.state_analytics.state_history_repository import StateHistoryRepository


class RecoveryForecaster:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        transitions: TransitionProbabilityEngine | None = None,
    ) -> None:
        self._config = config or DEFAULT_CONFIG
        self._transitions = transitions or TransitionProbabilityEngine(owns_connection=False)

    def forecast(self, context: GovernorContext) -> dict[str, Any]:
        state = context.current_state.lower()
        dd_used = _dd_budget_used_pct(context)
        dd_trend = self._dd_trend()
        base_risk = min(1.0, (context.risk_score / 100.0) * 0.5 + (dd_used / 100.0) * 0.5)

        probs: dict[str, float] = {}
        for days in self._config.forecast_days:
            if state == "recovery":
                exit_prob = self._transitions.cumulative_transition("recovery", "funded", steps=max(1, days // 30))
                probs[f"recovery_probability_{days}d"] = round(min(1.0, exit_prob + 0.1), 3)
            else:
                enter_prob = self._transition_enter_recovery(context, days)
                stress = base_risk + max(0.0, dd_trend) * 0.3
                probs[f"recovery_probability_{days}d"] = round(min(1.0, enter_prob * (0.5 + stress)), 3)

        return {
            "current_state": state,
            "recovery_probability": probs,
            "dd_trend": round(dd_trend, 4),
            "dd_budget_used_pct": round(dd_used, 2),
        }

    def _transition_enter_recovery(self, context: GovernorContext, days: int) -> float:
        state = context.current_state.lower()
        steps = max(1, days // 30)
        if state == "funded":
            return self._transitions.cumulative_transition("funded", "recovery", steps=steps)
        if state == "live":
            return self._transitions.cumulative_transition("live", "recovery", steps=steps)
        if state == "challenge":
            return self._transitions.cumulative_transition("challenge", "recovery", steps=steps) * 0.3
        return 0.05

    def _dd_trend(self) -> float:
        repo = StateHistoryRepository(owns_connection=False)
        try:
            rows = repo.list_recent(limit=30)
        finally:
            repo.close()
        dds = [float(r.get("drawdown_pct") or 0) for r in rows if r.get("drawdown_pct") is not None]
        if len(dds) < 2:
            return 0.0
        return (dds[-1] - dds[0]) / max(1, len(dds) - 1)
