"""AGE v3 Predictive Governor service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ai_governor_v3.engine import PredictiveGovernorEngine
from src.services.ai_governor_service import AiGovernorService


class PredictiveGovernorService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._age_v2_svc = AiGovernorService(owns_connections=owns_connections)
        self._engine = PredictiveGovernorEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._age_v2_svc.close()

    def _load_context_and_v2(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        persist_v2: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        ctx, profile_dict, prae_v2, state_summary, paae, pdts, slm, _ = self._age_v2_svc._load_upstream(
            source_path=source_path,
            profile_id=profile_id,
        )
        context = self._engine._age_v2.load_context(
            profile_ctx=profile_dict,
            prae_v2=prae_v2,
            state_analytics=state_summary,
            paae=paae,
            pdts=pdts,
            slm=slm,
        )
        v2_report = self._age_v2_svc.get_governor_report(
            source_path=source_path,
            profile_id=profile_id,
            persist=persist_v2,
        )
        return context, v2_report

    def get_predictive_governor_report(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        context, v2_report = self._load_context_and_v2(
            source_path=source_path,
            profile_id=profile_id,
            persist_v2=False,
        )
        return self._engine.run_predictive_cycle(context, age_v2_report=v2_report, persist=persist)

    def get_health_forecast(self, **kwargs: Any) -> dict[str, Any]:
        self.get_predictive_governor_report(**kwargs, persist=False)
        return self._engine.get_health_forecast()

    def get_recovery_forecast(self, **kwargs: Any) -> dict[str, Any]:
        self.get_predictive_governor_report(**kwargs, persist=False)
        return self._engine.get_recovery_forecast()

    def get_profile_transition_forecast(self, **kwargs: Any) -> dict[str, Any]:
        self.get_predictive_governor_report(**kwargs, persist=False)
        return self._engine.get_profile_transition_forecast()

    def get_strategy_forecast(self, **kwargs: Any) -> dict[str, Any]:
        self.get_predictive_governor_report(**kwargs, persist=False)
        return self._engine.get_strategy_forecast()

    def get_predictive_alerts(self, *, limit: int = 50, **kwargs: Any) -> list[dict[str, Any]]:
        self.get_predictive_governor_report(**kwargs, persist=False)
        return self._engine.get_predictive_alerts(limit=limit)

    def run_predictive_cycle(self, **kwargs: Any) -> dict[str, Any]:
        context, v2_report = self._load_context_and_v2(**kwargs, persist_v2=True)
        return self._engine.run_predictive_cycle(context, age_v2_report=v2_report, persist=True)

    def get_forecast_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._engine.get_forecast_history(limit=limit)
