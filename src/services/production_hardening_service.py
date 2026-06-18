"""Portfolio OS RC1 production hardening service."""
from __future__ import annotations

from typing import Any

from src.production_hardening.engine import ProductionHardeningEngine
from src.services.ai_cio_service import AiCioService
from src.services.cio_intelligence_service import CioIntelligenceService
from src.services.orl_service import OrlService


class ProductionHardeningService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._ai_cio = AiCioService(owns_connections=owns_connections)
        self._cil = CioIntelligenceService(owns_connections=owns_connections)
        self._orl = OrlService(owns_connections=owns_connections)
        self._engine = ProductionHardeningEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._orl.close()

    def _load_chain_context(self, **kwargs: Any) -> tuple[dict[str, Any], str]:
        cil, apm_v2, profile_id = self._cil.get_cil_and_apm_v2_reports(**kwargs)
        ai_cio = self._ai_cio.get_cio_report(**kwargs)
        orl = self._orl.get_operational_readiness(**kwargs)

        filter_kwargs = {k: v for k, v in kwargs.items() if k in ("source_path", "profile_id")}
        from src.api.apm_api import get_executive_status
        from src.api.cace_v16_api import get_confidence_intelligence_v16

        apm = get_executive_status(**filter_kwargs)
        cace = get_confidence_intelligence_v16(**filter_kwargs)

        upstream = self._build_upstream_from_cil(cil)
        confidence = cil.get("confidence_report") or {}
        age_report = {"recommended_action": confidence.get("recommended_action") or "DO_NOTHING"}

        ctx = {
            "profile_id": profile_id,
            "prae_report": upstream.get("prae_v2"),
            "paae_report": upstream.get("paae"),
            "pdts_report": upstream.get("pdts"),
            "slm_report": upstream.get("slm"),
            "age_report": age_report,
            "cace_report": cace,
            "mie_report": upstream.get("mie"),
            "apm_report": apm,
            "cil_report": cil,
            "ai_cio_report": ai_cio,
            "orl_report": orl,
        }
        return ctx, str(profile_id)

    def _build_upstream_from_cil(self, cil: dict[str, Any]) -> dict[str, Any]:
        layers = cil.get("layers") or {}
        return {
            "prae_v2": {"present": layers.get("prae")},
            "paae": {"present": layers.get("paae"), "current_weights": {}},
            "pdts": {"present": layers.get("pdts")},
            "slm": {"present": layers.get("slm")},
            "mie": {"present": layers.get("mie")},
        }

    def get_production_readiness(self, **kwargs: Any) -> dict[str, Any]:
        ctx, profile_id = self._load_chain_context(**kwargs)
        return self._engine.run(profile_id=profile_id, chain_context=ctx, persist=False, use_cache=True)

    def get_resilience_score(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_production_readiness(**kwargs)
        return {
            "profile_id": report.get("profile_id"),
            "resilience_score": report.get("resilience_score"),
            "resilience_status": report.get("resilience_status"),
            "resilience_components": report.get("resilience_components"),
            "production_ready": report.get("production_ready"),
        }

    def get_validation_results(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_production_readiness(**kwargs)
        return {
            "profile_id": report.get("profile_id"),
            "validation_results": report.get("validation_results"),
            "end_to_end": report.get("end_to_end"),
            "recommendation_chain": report.get("recommendation_chain"),
            "open_production_issues": report.get("open_production_issues"),
        }

    def get_benchmark_results(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_production_readiness(**kwargs)
        profile_id = str(report.get("profile_id") or "FundedBalanced")
        return {
            "profile_id": profile_id,
            "benchmark_results": report.get("benchmark_results"),
            "performance": report.get("performance"),
            "history": self._engine._repo.load_benchmark_history(profile_id=profile_id),
        }

    def run_production_validation(self, **kwargs: Any) -> dict[str, Any]:
        self._orl.run_operational_cycle(**kwargs)
        ctx, profile_id = self._load_chain_context(**kwargs)
        return self._engine.run(profile_id=profile_id, chain_context=ctx, persist=True, use_cache=False)
