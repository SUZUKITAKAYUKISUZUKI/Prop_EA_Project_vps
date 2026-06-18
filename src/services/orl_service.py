"""Operational Readiness Layer v1 service."""
from __future__ import annotations

from typing import Any

from src.orl.engine import OrlEngine
from src.orl.operational_api import extract_audit, extract_health
from src.services.ai_cio_service import AiCioService
from src.services.cio_intelligence_service import CioIntelligenceService


class OrlService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._ai_cio = AiCioService(owns_connections=owns_connections)
        self._cil = CioIntelligenceService(owns_connections=owns_connections)
        self._engine = OrlEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._ai_cio.close()

    def _load_context(self, **kwargs: Any) -> dict[str, Any]:
        cil, _apm_v2, profile_id = self._cil.get_cil_and_apm_v2_reports(**kwargs)
        ai_cio_report = self._ai_cio.get_cio_report(**kwargs)

        from src.api.apm_api import get_executive_status
        from src.api.cace_v16_api import get_confidence_intelligence_v16

        filter_kwargs = {k: v for k, v in kwargs.items() if k in ("source_path", "profile_id")}
        apm_report = get_executive_status(**filter_kwargs)
        cace_report = get_confidence_intelligence_v16(**filter_kwargs)

        confidence = cil.get("confidence_report") or {}
        opportunity = cil.get("opportunity_report") or {}
        age_report: dict[str, Any] = {}
        if confidence:
            age_report["recommended_action"] = confidence.get("recommended_action") or "DO_NOTHING"
        if opportunity:
            age_report["growth_signal"] = opportunity.get("growth_potential")

        return {
            "profile_id": profile_id,
            "ai_cio_report": ai_cio_report,
            "cil_report": cil,
            "apm_report": apm_report,
            "age_report": age_report,
            "cace_report": cace_report,
        }

    def get_operational_readiness(self, **kwargs: Any) -> dict[str, Any]:
        ctx = self._load_context(**kwargs)
        return self._engine.run(
            profile_id=ctx["profile_id"],
            ai_cio_report=ctx["ai_cio_report"],
            cil_report=ctx["cil_report"],
            apm_report=ctx["apm_report"],
            age_report=ctx["age_report"],
            cace_report=ctx["cace_report"],
            persist=False,
            use_cache=True,
        )

    def get_system_health(self, **kwargs: Any) -> dict[str, Any]:
        return extract_health(self.get_operational_readiness(**kwargs))

    def get_operational_audit(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_operational_readiness(**kwargs)
        return {
            **extract_audit(report),
            "audit_log": self._engine._repo.load_audit_log(profile_id=str(report.get("profile_id") or "FundedBalanced")),
        }

    def get_recommendation_consistency(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_operational_readiness(**kwargs)
        return dict(report.get("consistency") or {})

    def run_operational_cycle(self, **kwargs: Any) -> dict[str, Any]:
        self._ai_cio.run_cio_cycle(**kwargs)
        ctx = self._load_context(**kwargs)
        return self._engine.run(
            profile_id=ctx["profile_id"],
            ai_cio_report=ctx["ai_cio_report"],
            cil_report=ctx["cil_report"],
            apm_report=ctx["apm_report"],
            age_report=ctx["age_report"],
            cace_report=ctx["cace_report"],
            persist=True,
            use_cache=False,
        )
