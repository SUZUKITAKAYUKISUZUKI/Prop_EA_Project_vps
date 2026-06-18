"""AI CIO v1 service layer."""
from __future__ import annotations

from typing import Any

from src.ai_cio.cio_api import extract_memory, extract_opinion
from src.ai_cio.engine import AiCioEngine
from src.services.cio_intelligence_service import CioIntelligenceService


class AiCioService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._cil = CioIntelligenceService(owns_connections=owns_connections)
        self._engine = AiCioEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._cil.close()

    def _load_reports(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
        cil, apm_v2 = self._cil.get_cil_and_apm_v2_reports(**kwargs)
        profile_id = str(cil.get("profile_id") or apm_v2.get("profile_id") or "FundedBalanced")
        return cil, apm_v2, profile_id

    def get_cio_report(self, **kwargs: Any) -> dict[str, Any]:
        cil, apm_v2, profile_id = self._load_reports(**kwargs)
        return self._engine.run(
            profile_id=profile_id,
            cil_report=cil,
            apm_v2_report=apm_v2,
            persist=False,
            use_cache=True,
        )

    def get_cio_opinion(self, **kwargs: Any) -> dict[str, Any]:
        return extract_opinion(self.get_cio_report(**kwargs))

    def get_cio_recommendations(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_cio_report(**kwargs)
        return {
            "profile_id": report.get("profile_id"),
            "recommendations": report.get("recommendations"),
            "recommended_actions": report.get("recommended_actions"),
            "top_recommendation": report.get("top_recommendation"),
        }

    def get_cio_history(self, **kwargs: Any) -> dict[str, Any]:
        _, _, profile_id = self._load_reports(**kwargs)
        return {
            "profile_id": profile_id,
            "opinion_history": self._engine._repo.load_opinion_history(profile_id=profile_id),
            "recommendation_history": self._engine._repo.load_recommendation_history(profile_id=profile_id),
            "latest_report": self._engine._repo.load_latest_report(profile_id=profile_id),
        }

    def get_cio_memory(self, **kwargs: Any) -> dict[str, Any]:
        return extract_memory(self.get_cio_report(**kwargs))

    def run_cio_cycle(self, **kwargs: Any) -> dict[str, Any]:
        self._cil.run_cio_intelligence_cycle(**kwargs)
        cil, apm_v2 = self._cil.get_cil_and_apm_v2_reports(**kwargs, use_cache=True)
        profile_id = str(cil.get("profile_id") or apm_v2.get("profile_id") or "FundedBalanced")
        return self._engine.run(
            profile_id=profile_id,
            cil_report=cil,
            apm_v2_report=apm_v2,
            persist=True,
            use_cache=False,
        )
