"""RC2 Live Operations service layer."""
from __future__ import annotations

from typing import Any

from src.live_operations.engine import LiveOperationsEngine
from src.services.ai_cio_service import AiCioService
from src.services.cio_intelligence_service import CioIntelligenceService
from src.services.orl_service import OrlService
from src.services.production_hardening_service import ProductionHardeningService


class LiveOperationsService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._ai_cio = AiCioService(owns_connections=owns_connections)
        self._cil = CioIntelligenceService(owns_connections=owns_connections)
        self._orl = OrlService(owns_connections=owns_connections)
        self._production = ProductionHardeningService(owns_connections=owns_connections)
        self._engine = LiveOperationsEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._production.close()

    def _load_context(self, **kwargs: Any) -> tuple[dict[str, Any], str]:
        cil, _apm_v2, profile_id = self._cil.get_cil_and_apm_v2_reports(**kwargs)
        ai_cio = self._ai_cio.get_cio_report(**kwargs)
        orl = self._orl.get_operational_readiness(**kwargs)
        production = self._production.get_production_readiness(**kwargs)
        return (
            {
                "profile_id": profile_id,
                "ai_cio_report": ai_cio,
                "cil_report": cil,
                "orl_report": orl,
                "production_report": production,
            },
            str(profile_id),
        )

    def get_daily_briefing(self, **kwargs: Any) -> dict[str, Any]:
        report = self._run_read(**kwargs)
        return dict(report.get("daily_briefing") or {})

    def get_daily_digest(self, **kwargs: Any) -> dict[str, Any]:
        report = self._run_read(**kwargs)
        return {
            "daily_digest": report.get("daily_digest"),
            "digest_summary": report.get("digest_summary"),
        }

    def get_operational_alerts(self, **kwargs: Any) -> dict[str, Any]:
        report = self._run_read(**kwargs)
        _, profile_id = self._load_context(**kwargs)
        return {
            "profile_id": profile_id,
            "operational_alerts": report.get("operational_alerts"),
            "open_alerts": report.get("open_alerts"),
            "alerts_by_level": report.get("alerts_by_level"),
            "history": self._engine._repo.load_alerts(profile_id=profile_id),
        }

    def get_operational_score(self, **kwargs: Any) -> dict[str, Any]:
        report = self._run_read(**kwargs)
        return {
            "profile_id": report.get("profile_id"),
            "operational_score": report.get("operational_score"),
            "operational_components": report.get("operational_components"),
            "user_action_load": report.get("user_action_load"),
            "user_action_load_status": report.get("user_action_load_status"),
            "ai_cio_availability": report.get("ai_cio_availability"),
            "system_health": report.get("system_health"),
        }

    def get_live_readiness(self, **kwargs: Any) -> dict[str, Any]:
        report = self._run_read(**kwargs)
        return {
            "profile_id": report.get("profile_id"),
            "live_readiness": report.get("live_readiness"),
            "live_readiness_status": report.get("live_readiness_status"),
            "portfolio_os_complete": report.get("portfolio_os_complete"),
            "rc2_passed": report.get("rc2_passed"),
            "definition_of_done": report.get("definition_of_done"),
        }

    def run_live_operations_cycle(self, **kwargs: Any) -> dict[str, Any]:
        ctx, profile_id = self._load_context(**kwargs)
        return self._engine.run(profile_id=profile_id, ops_context=ctx, persist=True, use_cache=False)

    def _run_read(self, **kwargs: Any) -> dict[str, Any]:
        ctx, profile_id = self._load_context(**kwargs)
        return self._engine.run(profile_id=profile_id, ops_context=ctx, persist=False, use_cache=True)
