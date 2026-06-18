"""Risk budget forecast for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.risk_guardian import _dd_budget_used_pct
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG
from src.state_analytics.state_history_repository import StateHistoryRepository


class RiskBudgetForecaster:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG

    def forecast(self, context: GovernorContext) -> dict[str, Any]:
        dd_used = _dd_budget_used_pct(context)
        remaining = max(0.0, 100.0 - dd_used)
        budget_limit = float(context.profile.get("total_dd_limit") or 8.5)
        current_dd = context.dd_pct
        consumption_rate = self._consumption_rate_per_day(context)

        projections: dict[str, float] = {"current": round(remaining, 2)}
        dd_projections: dict[str, float] = {"current": round(current_dd, 2)}
        alerts: list[str] = []

        for days in self._config.forecast_days:
            projected_remaining = max(0.0, remaining - consumption_rate * days)
            burn = max(0.0, remaining - projected_remaining)
            projected_dd = min(100.0, current_dd + burn * 0.05)
            projections[f"{days}d"] = round(projected_remaining, 2)
            dd_projections[f"{days}d"] = round(projected_dd, 2)
            if projected_remaining < self._config.risk_budget_exhaustion_threshold:
                alerts.append(f"RISK_BUDGET_EXHAUSTION_{days}d")

        return {
            "current_dd_pct": round(current_dd, 2),
            "dd_budget_limit": budget_limit,
            "risk_budget_remaining_pct": round(remaining, 2),
            "risk_budget_forecast": projections,
            "dd_forecast": dd_projections,
            "consumption_rate_per_day": round(consumption_rate, 4),
            "alerts": alerts,
        }

    def _consumption_rate_per_day(self, context: GovernorContext) -> float:
        repo = StateHistoryRepository(owns_connection=False)
        try:
            rows = repo.list_recent(limit=20)
        finally:
            repo.close()
        budgets = [
            float(r.get("risk_budget_remaining") or 0)
            for r in rows
            if r.get("risk_budget_remaining") is not None
        ]
        if len(budgets) >= 2:
            return max(0.0, (budgets[0] - budgets[-1]) / max(1, len(budgets) - 1))

        dd_used = _dd_budget_used_pct(context)
        risk_factor = context.risk_score / 100.0
        return round(0.05 + dd_used * 0.002 + risk_factor * 0.03, 4)
