"""CIO Intelligence Layer v1 service."""
from __future__ import annotations

from typing import Any

from src.cio_intelligence.engine import CioIntelligenceEngine
from src.services.apm_v2_service import ApmV2Service


class CioIntelligenceService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._apm_v2 = ApmV2Service(owns_connections=owns_connections)
        self._engine = CioIntelligenceEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._apm_v2.close()

    def _run_full_stack(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        upstream, apm_v1, v17, mie, v16 = self._apm_v2._run_stack(**kwargs)
        apm_v2 = self._apm_v2._engine.run(
            profile_id=upstream["profile_id"],
            apm_v1_report=apm_v1,
            cace_v17_report=v17,
            mie_report=mie,
            upstream=upstream,
            persist=kwargs.get("persist", False),
            use_cache=not kwargs.get("persist", False),
        )
        return upstream, v16, v17, mie, apm_v1, apm_v2, upstream["profile_id"]

    def get_cio_intelligence(self, **kwargs: Any) -> dict[str, Any]:
        upstream, v16, v17, mie, apm_v1, apm_v2, profile_id = self._run_full_stack(**kwargs)
        return self._engine.run(
            profile_id=profile_id,
            upstream=upstream,
            cace_v16_report=v16,
            cace_v17_report=v17,
            mie_report=mie,
            apm_v1_report=apm_v1,
            apm_v2_report=apm_v2,
            persist=False,
            use_cache=True,
        )

    def get_investment_state(self, **kwargs: Any) -> list[str]:
        return list(self.get_cio_intelligence(**kwargs).get("investment_state") or [])

    def get_executive_score(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_cio_intelligence(**kwargs)
        return {
            "executive_score": report.get("executive_score"),
            "components": report.get("executive_components"),
            "opportunity_score": report.get("opportunity_score"),
            "risk_score": report.get("risk_score"),
            "confidence_score": report.get("confidence_score"),
            "capital_efficiency": report.get("capital_efficiency"),
            "intelligence_trust": report.get("intelligence_trust"),
        }

    def get_opportunity_report(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.get_cio_intelligence(**kwargs).get("opportunity_report") or {})

    def get_risk_report(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.get_cio_intelligence(**kwargs).get("risk_report") or {})

    def get_capital_efficiency(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.get_cio_intelligence(**kwargs).get("capital_efficiency_report") or {})

    def get_cil_and_apm_v2_reports(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return CIL and APM v2 reports for AI CIO (read path)."""
        upstream, v16, v17, mie, apm_v1, apm_v2, profile_id = self._run_full_stack(**kwargs)
        cil = self._engine.run(
            profile_id=profile_id,
            upstream=upstream,
            cace_v16_report=v16,
            cace_v17_report=v17,
            mie_report=mie,
            apm_v1_report=apm_v1,
            apm_v2_report=apm_v2,
            persist=kwargs.get("persist", False),
            use_cache=kwargs.get("use_cache", not kwargs.get("persist", False)),
        )
        return cil, apm_v2

    def run_cio_intelligence_cycle(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._apm_v2._apm._mie._load_upstream(**{**kwargs, "persist_age_v4": True})
        v16, v17 = self._apm_v2._apm._mie._run_stack(upstream, persist=True)
        from src.meta_intelligence.engine import MetaIntelligenceEngine

        mie = MetaIntelligenceEngine(owns_connections=False).run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            cace_v17_report=v17,
            upstream=upstream,
            persist=True,
            use_cache=False,
        )
        apm_v1 = self._apm_v2._apm._engine.run(
            profile_id=upstream["profile_id"],
            upstream=upstream,
            cace_v16_report=v16,
            cace_v17_report=v17,
            mie_report=mie,
            persist=True,
            use_cache=False,
        )
        apm_v2 = self._apm_v2._engine.run(
            profile_id=upstream["profile_id"],
            apm_v1_report=apm_v1,
            cace_v17_report=v17,
            mie_report=mie,
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
            apm_v1_report=apm_v1,
            apm_v2_report=apm_v2,
            persist=True,
            use_cache=False,
        )
