"""CIL v1 orchestration — read-only executive investment view."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.cio_intelligence.capital_efficiency_engine import CapitalEfficiencyEngine
from src.cio_intelligence.confidence_engine import ConfidenceEngine
from src.cio_intelligence.config import (
    CACHE_CIO_INTELLIGENCE,
    CACHE_CIO_OPPORTUNITY,
    CACHE_CIO_RISK,
    CACHE_CIO_STATE,
)
from src.cio_intelligence.executive_summary_engine import ExecutiveSummaryEngine
from src.cio_intelligence.intelligence_aggregator import IntelligenceAggregator
from src.cio_intelligence.intelligence_report import IntelligenceReport
from src.cio_intelligence.intelligence_repository import IntelligenceRepository
from src.cio_intelligence.investment_state_engine import InvestmentStateEngine
from src.cio_intelligence.opportunity_engine import OpportunityEngine
from src.cio_intelligence.risk_engine import RiskEngine


class CioIntelligenceEngine:
    """Transforms Portfolio OS intelligence into a single executive investment view."""

    def __init__(
        self,
        *,
        aggregator: IntelligenceAggregator | None = None,
        state_engine: InvestmentStateEngine | None = None,
        opportunity_engine: OpportunityEngine | None = None,
        risk_engine: RiskEngine | None = None,
        confidence_engine: ConfidenceEngine | None = None,
        capital_engine: CapitalEfficiencyEngine | None = None,
        summary_engine: ExecutiveSummaryEngine | None = None,
        repo: IntelligenceRepository | None = None,
        reporter: IntelligenceReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._aggregator = aggregator or IntelligenceAggregator()
        self._state = state_engine or InvestmentStateEngine()
        self._opportunity = opportunity_engine or OpportunityEngine()
        self._risk = risk_engine or RiskEngine()
        self._confidence = confidence_engine or ConfidenceEngine()
        self._capital = capital_engine or CapitalEfficiencyEngine()
        self._summary = summary_engine or ExecutiveSummaryEngine()
        self._repo = repo or IntelligenceRepository(owns_connection=self._owns)
        self._reporter = reporter or IntelligenceReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        upstream: dict[str, Any],
        cace_v16_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        mie_report: dict[str, Any],
        apm_v1_report: dict[str, Any],
        apm_v2_report: dict[str, Any] | None = None,
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_CIO_INTELLIGENCE.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        bundle = self._aggregator.aggregate(
            upstream=upstream,
            cace_v16_report=cace_v16_report,
            cace_v17_report=cace_v17_report,
            mie_report=mie_report,
            apm_v1_report=apm_v1_report,
            apm_v2_report=apm_v2_report,
        )
        investment_states = self._state.evaluate(bundle)
        opportunity = self._opportunity.evaluate(bundle)
        risk = self._risk.evaluate(bundle)
        confidence = self._confidence.evaluate(bundle)
        capital_efficiency = self._capital.evaluate(bundle)
        summary = self._summary.evaluate(
            investment_states=investment_states,
            opportunity=opportunity,
            risk=risk,
            confidence=confidence,
            capital_efficiency=capital_efficiency,
            bundle=bundle,
        )

        report = self._reporter.build(
            profile_id=profile_id,
            summary=summary,
            opportunity=opportunity,
            risk=risk,
            confidence=confidence,
            capital_efficiency=capital_efficiency,
            bundle=bundle,
        )

        if persist:
            self._repo.save_snapshot(profile_id=profile_id, report=report)
            self._repo.save_executive_score(
                profile_id=profile_id,
                executive_score=float(report.get("executive_score") or 0),
                opportunity_score=float(report.get("opportunity_score") or 0),
                risk_score=float(report.get("risk_score") or 0),
                confidence_score=float(report.get("confidence_score") or 0),
                capital_efficiency=float(report.get("capital_efficiency") or 0),
            )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_CIO_STATE.format(profile_id=profile_id), investment_states)
            self._cache.set(CACHE_CIO_OPPORTUNITY.format(profile_id=profile_id), opportunity)
            self._cache.set(CACHE_CIO_RISK.format(profile_id=profile_id), risk)

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}
