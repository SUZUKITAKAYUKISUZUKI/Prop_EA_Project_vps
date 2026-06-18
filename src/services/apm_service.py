"""APM v1 service layer."""
from __future__ import annotations

from typing import Any

from src.apm.engine import ApmEngine
from src.meta_intelligence.engine import MetaIntelligenceEngine
from src.services.mie_service import MieService


class ApmService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._mie = MieService(owns_connections=owns_connections)
        self._engine = ApmEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._mie.close()

    def _run_intelligence_stack(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        upstream = self._mie._load_upstream(**kwargs)
        v16, v17 = self._mie._run_stack(upstream, persist=kwargs.get("persist", False))
        mie = MetaIntelligenceEngine(owns_connections=False).run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            cace_v17_report=v17,
            upstream=upstream,
            persist=kwargs.get("persist", False),
            use_cache=not kwargs.get("persist", False),
        )
        return upstream, v16, v17, mie

    def get_executive_status(self, **kwargs: Any) -> dict[str, Any]:
        upstream, v16, v17, mie = self._run_intelligence_stack(**kwargs)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            upstream=upstream,
            cace_v16_report=v16,
            cace_v17_report=v17,
            mie_report=mie,
            persist=False,
            use_cache=True,
        )

    def get_execution_queue(self, **kwargs: Any) -> list[dict[str, Any]]:
        report = self.get_executive_status(**kwargs)
        return list(report.get("execution_queue") or report.get("approval_queue") or [])

    def get_roadmap(self, **kwargs: Any) -> list[dict[str, Any]]:
        report = self.get_executive_status(**kwargs)
        return list(report.get("roadmap") or [])

    def get_opportunities(self, **kwargs: Any) -> list[dict[str, Any]]:
        report = self.get_executive_status(**kwargs)
        return list(report.get("opportunities") or [])

    def get_risk_alerts(self, **kwargs: Any) -> list[dict[str, Any]]:
        report = self.get_executive_status(**kwargs)
        return list(report.get("risk_alerts") or [])

    def approve_action(self, *, action_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._engine.approve_action(action_id=action_id)

    def reject_action(self, *, action_id: str, reason: str = "", **kwargs: Any) -> dict[str, Any]:
        return self._engine.reject_action(action_id=action_id, reason=reason)

    def run_apm_cycle(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._mie._load_upstream(**{**kwargs, "persist_age_v4": True})
        v16, v17 = self._mie._run_stack(upstream, persist=True)
        mie = MetaIntelligenceEngine(owns_connections=False).run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            cace_v17_report=v17,
            upstream=upstream,
            persist=True,
            use_cache=False,
        )
        return self._engine.run(
            profile_id=upstream["profile_id"],
            upstream=upstream,
            cace_v16_report=v16,
            cace_v17_report=v17,
            mie_report=mie,
            persist=True,
            use_cache=False,
        )
