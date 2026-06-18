"""CACE service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.cace.engine import CaceEngine
from src.services.strategic_governor_service import StrategicGovernorService


class CaceService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._strategic_svc = StrategicGovernorService(owns_connections=owns_connections)
        self._engine = CaceEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._strategic_svc.close()

    def _load_upstream_and_age_v4(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        persist_age_v4: bool = False,
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        context, v2_report = self._strategic_svc._load_context_and_v2(
            source_path=source_path,
            profile_id=profile_id,
            persist_v2=False,
        )
        age_v4 = self._strategic_svc._engine.run_strategic_cycle(
            context,
            age_v2_report=v2_report,
            persist=persist_age_v4,
        )
        pid = context.profile_id
        return (
            pid,
            context.paae,
            context.pdts,
            context.prae_v2,
            context.state_analytics,
            context.slm,
            age_v4,
        )

    def run_confidence_cycle(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        pid, paae, pdts, prae_v2, state, slm, age_v4 = self._load_upstream_and_age_v4(
            source_path=source_path,
            profile_id=profile_id,
            persist_age_v4=True,
        )
        report = self._engine.run_confidence_cycle(
            profile_id=pid,
            paae=paae,
            pdts=pdts,
            prae_v2=prae_v2,
            state_analytics=state,
            slm=slm,
            age_v4=age_v4,
            current_weights=None,
            persist=True,
            enrich_age_v4=True,
        )
        report["age_v4"] = report.get("age_v4_enriched") or age_v4
        return report

    def get_confidence_report(self, **kwargs: Any) -> dict[str, Any]:
        pid, paae, pdts, prae_v2, state, slm, age_v4 = self._load_upstream_and_age_v4(**kwargs, persist_age_v4=False)
        return self._engine.run_confidence_cycle(
            profile_id=pid,
            paae=paae,
            pdts=pdts,
            prae_v2=prae_v2,
            state_analytics=state,
            slm=slm,
            age_v4=age_v4,
            persist=False,
            enrich_age_v4=True,
        )

    def get_allocation_confidence(self, **kwargs: Any) -> dict[str, Any]:
        self.get_confidence_report(**kwargs)
        return self._engine.get_allocation_confidence()

    def get_strategy_confidence(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.get_confidence_report(**kwargs)
        return self._engine.get_strategy_confidence()

    def get_portfolio_confidence(self, **kwargs: Any) -> dict[str, Any]:
        self.get_confidence_report(**kwargs)
        return self._engine.get_portfolio_confidence()

    def get_confidence_history(self, *, profile_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        return self._engine.get_confidence_history(profile_id=profile_id, limit=limit)
