"""Portfolio health forecast for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.decision_history import DecisionHistory
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG


class HealthForecaster:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG

    def forecast(
        self,
        context: GovernorContext,
        *,
        health_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        current = float(context.health_score)
        slope = self._daily_slope(health_history or [])
        if slope == 0.0:
            slope = self._heuristic_slope(context)

        projections: dict[str, float] = {"current": round(current, 1)}
        for days in self._config.forecast_days:
            projected = max(0.0, min(100.0, current + slope * days))
            projections[f"{days}d"] = round(projected, 1)

        return {
            "current_health": round(current, 1),
            "future_health": projections,
            "health_forecast_json": projections,
            "daily_slope": round(slope, 4),
        }

    def _daily_slope(self, history: list[dict[str, Any]]) -> float:
        scores = [float(r.get("health_score") or 0) for r in history if r.get("health_score") is not None]
        if len(scores) < 2:
            return 0.0
        # Use last up to 14 points; assume ~1 day between governor snapshots
        window = scores[-14:]
        return (window[-1] - window[0]) / max(1, len(window) - 1)

    def _heuristic_slope(self, context: GovernorContext) -> float:
        dd_pressure = context.dd_pct / 100.0
        risk_pressure = context.risk_score / 100.0
        recovery_drag = context.recovery_events * 0.15
        return round(-(dd_pressure * 0.12 + risk_pressure * 0.08 + recovery_drag), 4)

    def load_health_history(self, profile_id: str) -> list[dict[str, Any]]:
        history = DecisionHistory(owns_connection=True)
        try:
            return history.health_history(limit=30)
        finally:
            history.close()
