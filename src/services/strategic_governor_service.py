"""AGE v4 Strategic Governor service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ai_governor_v4.engine import StrategicGovernorEngine
from src.services.predictive_governor_service import PredictiveGovernorService


class StrategicGovernorService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._age_v3_svc = PredictiveGovernorService(owns_connections=owns_connections)
        self._engine = StrategicGovernorEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._age_v3_svc.close()

    def _load_context_and_v2(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        persist_v2: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        return self._age_v3_svc._load_context_and_v2(
            source_path=source_path,
            profile_id=profile_id,
            persist_v2=persist_v2,
        )

    def get_strategic_report(
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
        return self._engine.run_strategic_cycle(context, age_v2_report=v2_report, persist=persist)

    def get_future_scenarios(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.get_strategic_report(**kwargs, persist=False)
        return self._engine.get_future_scenarios()

    def get_future_rankings(self, **kwargs: Any) -> dict[str, Any]:
        self.get_strategic_report(**kwargs, persist=False)
        return self._engine.get_future_rankings()

    def get_best_future(self, **kwargs: Any) -> dict[str, Any]:
        self.get_strategic_report(**kwargs, persist=False)
        return self._engine.get_best_future()

    def run_strategic_cycle(self, **kwargs: Any) -> dict[str, Any]:
        context, v2_report = self._load_context_and_v2(**kwargs, persist_v2=True)
        return self._engine.run_strategic_cycle(context, age_v2_report=v2_report, persist=True)

    def get_scenario_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._engine.get_scenario_history(limit=limit)
