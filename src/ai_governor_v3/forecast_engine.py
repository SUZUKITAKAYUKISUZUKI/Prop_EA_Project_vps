"""Orchestrates all AGE v3 forecast categories."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.allocation_forecaster import AllocationForecaster
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG
from src.ai_governor_v3.health_forecaster import HealthForecaster
from src.ai_governor_v3.lifecycle_forecaster import LifecycleForecaster
from src.ai_governor_v3.profile_forecaster import ProfileForecaster
from src.ai_governor_v3.recovery_forecaster import RecoveryForecaster
from src.ai_governor_v3.risk_budget_forecaster import RiskBudgetForecaster
from src.ai_governor_v3.transition_probability_engine import TransitionProbabilityEngine


class ForecastEngine:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        transitions: TransitionProbabilityEngine | None = None,
    ) -> None:
        self._config = config or DEFAULT_CONFIG
        self._transitions = transitions or TransitionProbabilityEngine(owns_connection=False)
        self._health = HealthForecaster(self._config)
        self._recovery = RecoveryForecaster(self._config, self._transitions)
        self._risk_budget = RiskBudgetForecaster(self._config)
        self._lifecycle = LifecycleForecaster(self._config)
        self._allocation = AllocationForecaster(self._config)
        self._profile = ProfileForecaster(self._config, self._transitions)

    def close(self) -> None:
        self._transitions.close()

    def run_all(self, context: GovernorContext) -> dict[str, Any]:
        health_history = self._health.load_health_history(context.profile_id)
        return {
            "health": self._health.forecast(context, health_history=health_history),
            "recovery": self._recovery.forecast(context),
            "risk_budget": self._risk_budget.forecast(context),
            "lifecycle": self._lifecycle.forecast(context),
            "allocation": self._allocation.forecast(context),
            "profile": self._profile.forecast(context),
            "transitions": self._transitions.to_dict(),
            "forecast_horizons": list(self._config.forecast_days),
        }
