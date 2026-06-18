"""Core predictive cycle orchestration for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.deterioration_detector import DeteriorationDetector
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG
from src.ai_governor_v3.forecast_engine import ForecastEngine
from src.ai_governor_v3.forecast_repository import ForecastRepository
from src.ai_governor_v3.future_state_simulator import FutureStateSimulator
from src.ai_governor_v3.predictive_recommendation_engine import PredictiveRecommendationEngine
from src.ai_governor_v3.predictive_report import PredictiveReport
from src.ai_governor_v3.scenario_projection_engine import ScenarioProjectionEngine
from src.repositories.cache_repository import CacheRepository


class PredictiveEngine:
    def __init__(
        self,
        *,
        config: ForecastConfig | None = None,
        forecast_engine: ForecastEngine | None = None,
        simulator: FutureStateSimulator | None = None,
        scenario: ScenarioProjectionEngine | None = None,
        detector: DeteriorationDetector | None = None,
        recommendations: PredictiveRecommendationEngine | None = None,
        reporter: PredictiveReport | None = None,
        repo: ForecastRepository | None = None,
        cache: CacheRepository | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._config = config or DEFAULT_CONFIG
        self._owns = owns_connections or repo is None
        self._forecast = forecast_engine or ForecastEngine(self._config)
        self._simulator = simulator or FutureStateSimulator(self._config)
        self._scenario = scenario or ScenarioProjectionEngine()
        self._detector = detector or DeteriorationDetector(self._config)
        self._recommendations = recommendations or PredictiveRecommendationEngine()
        self._reporter = reporter or PredictiveReport()
        self._repo = repo or ForecastRepository(owns_connection=self._owns)
        self._cache = cache or CacheRepository(owns_connection=False)

    def close(self) -> None:
        self._forecast.close()
        if self._owns:
            self._repo.close()

    def run(
        self,
        context: GovernorContext,
        *,
        age_v2_report: dict[str, Any],
        persist: bool = True,
    ) -> dict[str, Any]:
        cache_key = f"age_v3_forecast:{context.profile_id}"
        if self._config.use_cache:
            cached = self._cache.get(cache_key)
            if cached and not persist:
                return cached

        forecasts = self._forecast.run_all(context)
        timeline = self._simulator.simulate_future_state(context, forecasts=forecasts)
        scenario_projection = self._scenario.project(context)
        alerts = self._detector.detect(context, forecasts)
        recommendations = self._recommendations.build(
            context,
            forecasts,
            alerts,
            age_v2_report=age_v2_report,
        )
        report = self._reporter.build(
            forecasts=forecasts,
            timeline=timeline,
            scenario_projection=scenario_projection,
            alerts=alerts,
            recommendations=recommendations,
            age_v2_report=age_v2_report,
            profile_id=context.profile_id,
        )

        if persist:
            self._repo.save_forecast(
                forecast_horizon=self._config.horizons_label(),
                health_forecast=forecasts.get("health") or {},
                risk_forecast=forecasts.get("risk_budget") or {},
                recovery_probability=forecasts.get("recovery") or {},
                future_state=timeline,
                confidence=float(report.get("confidence") or 0),
                recommendation_json=recommendations,
                profile_id=context.profile_id,
            )
            self._repo.save_alerts(alerts)

        if self._config.use_cache:
            self._cache.set(cache_key, report)

        return report
