"""AI CIO v1 orchestration — executive advisor only."""
from __future__ import annotations

from typing import Any

from src.ai_cio.cio_context import CioContext
from src.ai_cio.cio_executive_report_engine import CioExecutiveReportEngine
from src.ai_cio.cio_memory_engine import CioMemoryEngine
from src.ai_cio.cio_opinion_engine import CioOpinionEngine
from src.ai_cio.cio_opportunity_engine import CioOpportunityEngine
from src.ai_cio.cio_recommendation_engine import CioRecommendationEngine
from src.ai_cio.cio_report import CioReport
from src.ai_cio.cio_repository import CioRepository
from src.ai_cio.cio_risk_engine import CioRiskEngine
from src.ai_cio.config import (
    CACHE_AI_CIO_OPINION,
    CACHE_AI_CIO_RECOMMENDATIONS,
    CACHE_AI_CIO_REPORT,
)
from src.cace.confidence_cache import ConfidenceCache


class AiCioEngine:
    """Converts CIL + APM v2 intelligence into executive investment decisions."""

    def __init__(
        self,
        *,
        context_builder: CioContext | None = None,
        opinion_engine: CioOpinionEngine | None = None,
        memory_engine: CioMemoryEngine | None = None,
        opportunity_engine: CioOpportunityEngine | None = None,
        risk_engine: CioRiskEngine | None = None,
        recommendation_engine: CioRecommendationEngine | None = None,
        executive_engine: CioExecutiveReportEngine | None = None,
        repo: CioRepository | None = None,
        reporter: CioReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._context = context_builder or CioContext()
        self._opinion = opinion_engine or CioOpinionEngine()
        self._memory = memory_engine or CioMemoryEngine()
        self._opportunity = opportunity_engine or CioOpportunityEngine()
        self._risk = risk_engine or CioRiskEngine()
        self._recommendations = recommendation_engine or CioRecommendationEngine()
        self._executive = executive_engine or CioExecutiveReportEngine()
        self._repo = repo or CioRepository(owns_connection=self._owns)
        self._reporter = reporter or CioReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        cil_report: dict[str, Any],
        apm_v2_report: dict[str, Any],
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_AI_CIO_REPORT.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        context = self._context.build(cil_report=cil_report, apm_v2_report=apm_v2_report)
        memory = self._memory.evaluate(context)
        opinion = self._opinion.evaluate(context)
        opportunity = self._opportunity.evaluate(context)
        risk = self._risk.evaluate(context)
        recommendations = self._recommendations.evaluate(
            context,
            opinion=str(opinion.get("cio_opinion") or "MAINTAIN"),
            recommended_profile=str(memory.get("recommended_profile") or profile_id),
        )
        executive = self._executive.evaluate(context, opinion=opinion, memory=memory)

        report = self._reporter.build(
            profile_id=profile_id,
            executive=executive,
            opinion=opinion,
            opportunity=opportunity,
            risk=risk,
            memory=memory,
            recommendations=recommendations,
        )

        if persist:
            self._repo.save_report(profile_id=profile_id, report=report)
            self._repo.save_opinion(
                profile_id=profile_id,
                opinion=str(report.get("cio_opinion") or "MAINTAIN"),
                cio_score=float(report.get("cio_score") or 0),
            )
            self._repo.save_recommendations(
                profile_id=profile_id,
                recommendations=list(report.get("recommendations") or []),
            )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(
                CACHE_AI_CIO_OPINION.format(profile_id=profile_id),
                {
                    "cio_opinion": report.get("cio_opinion"),
                    "cio_score": report.get("cio_score"),
                },
            )
            self._cache.set(
                CACHE_AI_CIO_RECOMMENDATIONS.format(profile_id=profile_id),
                report.get("recommendations"),
            )

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}
