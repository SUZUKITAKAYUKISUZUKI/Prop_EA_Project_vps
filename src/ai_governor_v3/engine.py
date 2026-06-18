"""AGE v3 Predictive Governor — top-level engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ai_governor.engine import AiGovernorEngine
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG
from src.ai_governor_v3.forecast_repository import ForecastRepository
from src.ai_governor_v3.predictive_engine import PredictiveEngine


class PredictiveGovernorEngine:
    """AGE v3 extends AGE v2 with predictive governance — never places trades."""

    def __init__(
        self,
        *,
        age_v2: AiGovernorEngine | None = None,
        predictive: PredictiveEngine | None = None,
        repo: ForecastRepository | None = None,
        config: ForecastConfig | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._config = config or DEFAULT_CONFIG
        self._age_v2 = age_v2 or AiGovernorEngine(owns_connections=owns_connections)
        self._predictive = predictive or PredictiveEngine(config=self._config, owns_connections=owns_connections)
        self._repo = repo or ForecastRepository(owns_connection=False)
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        self._predictive.close()
        if self._owns:
            self._age_v2.close()

    def run_predictive_cycle(
        self,
        context: GovernorContext,
        *,
        age_v2_report: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        v2_report = age_v2_report or self._age_v2.run_governor_cycle(context, persist=False)
        report = self._predictive.run(context, age_v2_report=v2_report, persist=persist)
        report["age_v2"] = v2_report
        self._last_report = report
        return report

    def get_predictive_governor_report(self) -> dict[str, Any]:
        return self._last_report or {}

    def get_health_forecast(self) -> dict[str, Any]:
        report = self.get_predictive_governor_report()
        return report.get("health_forecast") or {}

    def get_recovery_forecast(self) -> dict[str, Any]:
        report = self.get_predictive_governor_report()
        return report.get("recovery_forecast") or {}

    def get_profile_transition_forecast(self) -> dict[str, Any]:
        report = self.get_predictive_governor_report()
        return report.get("profile_transition_forecast") or {}

    def get_strategy_forecast(self) -> dict[str, Any]:
        report = self.get_predictive_governor_report()
        return report.get("strategy_forecast") or {}

    def get_predictive_alerts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if self._last_report:
            return self._last_report.get("predictive_alerts") or []
        return self._repo.list_alerts(limit=limit)

    def get_forecast_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._repo.list_forecasts(limit=limit)
