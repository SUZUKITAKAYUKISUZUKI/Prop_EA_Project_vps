"""Core CACE confidence orchestration."""
from __future__ import annotations

from typing import Any

from src.cace.allocation_confidence_engine import AllocationConfidenceEngine
from src.cace.confidence_factors import CACHE_KEY_ALLOCATION, CACHE_KEY_PORTFOLIO, CACHE_KEY_STRATEGY, ConfidenceFactors
from src.cace.confidence_cache import ConfidenceCache
from src.cace.confidence_report import ConfidenceReport
from src.cace.confidence_repository import ConfidenceRepository
from src.cace.forecast_confidence_engine import ForecastConfidenceEngine
from src.cace.historical_reliability_engine import HistoricalReliabilityEngine
from src.cace.monte_carlo_confidence_engine import MonteCarloConfidenceEngine
from src.cace.portfolio_confidence_engine import PortfolioConfidenceEngine
from src.cace.strategy_confidence_engine import StrategyConfidenceEngine


class ConfidenceEngine:
    """Evaluates allocation recommendation reliability — never modifies allocations."""

    def __init__(
        self,
        *,
        allocation: AllocationConfidenceEngine | None = None,
        strategy: StrategyConfidenceEngine | None = None,
        portfolio: PortfolioConfidenceEngine | None = None,
        reporter: ConfidenceReport | None = None,
        repo: ConfidenceRepository | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections or repo is None
        self._allocation = allocation or AllocationConfidenceEngine()
        self._strategy = strategy or StrategyConfidenceEngine()
        self._portfolio = portfolio or PortfolioConfidenceEngine()
        self._reporter = reporter or ConfidenceReport()
        self._repo = repo or ConfidenceRepository(owns_connection=self._owns)
        self._cache = cache or ConfidenceCache()
        self._historical = HistoricalReliabilityEngine()
        self._monte_carlo = MonteCarloConfidenceEngine()
        self._forecast = ForecastConfidenceEngine()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        paae: dict[str, Any],
        pdts: dict[str, Any],
        prae_v2: dict[str, Any],
        state_analytics: dict[str, Any],
        slm: dict[str, Any],
        age_v4: dict[str, Any],
        current_weights: dict[str, float] | None = None,
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        portfolio_key = CACHE_KEY_PORTFOLIO.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(portfolio_key)
            if cached:
                self._last_report = cached
                return cached

        weights = current_weights or paae.get("recommended_weights") or {}
        age_v3 = age_v4.get("age_v3") or {}

        alloc = self._allocation.evaluate(
            allocation_id="recommended",
            profile_id=profile_id,
            weights=weights,
            paae=paae,
            pdts=pdts,
            prae_v2=prae_v2,
            state_analytics=state_analytics,
            slm=slm,
            age_v4=age_v4,
            age_v3=age_v3,
        )

        factors = ConfidenceFactors(**alloc["factors"])
        breakdown = {
            "historical_reliability": factors.historical_reliability,
            "monte_carlo_stability": factors.monte_carlo_stability,
            "forecast_stability": factors.forecast_stability,
            "portfolio_health": factors.portfolio_health,
            "lifecycle_quality": factors.lifecycle_quality,
        }
        alloc["breakdown"] = {k: round(v, 1) for k, v in breakdown.items()}
        strategies = self._strategy.evaluate_all(
            paae=paae,
            prae_v2=prae_v2,
            slm=slm,
            state_analytics=state_analytics,
            portfolio_factors=factors,
        )
        portfolio = self._portfolio.evaluate(
            profile_id=profile_id,
            factors=factors,
            allocation_confidence=alloc,
            strategy_confidences=strategies,
            age_v4=age_v4,
            paae=paae,
        )

        history = self._repo.list_confidence_history(profile_id=profile_id, limit=10)
        report = self._reporter.build(
            profile_id=profile_id,
            allocation_confidence=alloc,
            strategy_confidences=strategies,
            portfolio_confidence=portfolio,
            confidence_history=history,
        )

        if persist:
            self._persist(profile_id, alloc, strategies, portfolio, report)

        if use_cache:
            self._cache.set(CACHE_KEY_PORTFOLIO.format(profile_id=profile_id), report)
            self._cache.set(CACHE_KEY_ALLOCATION.format(profile_id=profile_id), alloc)
            for row in strategies:
                code = str(row.get("strategy") or "")
                if code:
                    self._cache.set(CACHE_KEY_STRATEGY.format(strategy=code), row)

        self._last_report = report
        return report

    def enrich_age_v4_report(self, age_v4_report: dict[str, Any], cace_report: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(age_v4_report)
        if "confidence" in enriched and "strategic_confidence" not in enriched:
            enriched["strategic_confidence"] = enriched["confidence"]
        enriched["confidence"] = cace_report.get("confidence")
        enriched["confidence_category"] = cace_report.get("confidence_category")
        enriched["future_score"] = enriched.get("strategic_score") or enriched.get("future_score")
        enriched["cace"] = {
            "allocation_confidence": cace_report.get("allocation_confidence"),
            "portfolio_confidence": cace_report.get("portfolio_confidence"),
            "strategy_confidence": cace_report.get("strategy_confidence"),
            "top_risks": cace_report.get("top_risks"),
            "top_opportunities": cace_report.get("top_opportunities"),
        }
        return enriched

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}

    def _persist(
        self,
        profile_id: str,
        alloc: dict[str, Any],
        strategies: list[dict[str, Any]],
        portfolio: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        self._repo.save_allocation_confidence(
            profile_id=profile_id,
            allocation_json=alloc.get("allocation_json") or {},
            confidence=float(alloc.get("confidence") or 0),
            category=str(alloc.get("category") or "VERY_LOW"),
            expected_r=float(alloc.get("expected_r") or 0),
            expected_pf=float(alloc.get("expected_pf") or 0),
            expected_dd=float(alloc.get("expected_dd") or 0),
            reason_json=list(alloc.get("reason") or []),
            breakdown=alloc.get("breakdown") or alloc.get("factors"),
        )
        for row in strategies:
            self._repo.save_strategy_confidence(
                strategy=str(row.get("strategy") or ""),
                confidence=float(row.get("confidence") or 0),
                portfolio_fit=float(row.get("portfolio_fit") or 0),
                lifecycle_stage=str(row.get("lifecycle_stage") or ""),
                reason_json=list(row.get("reason") or []),
                breakdown=alloc.get("breakdown"),
            )
        self._repo.save_confidence_history(
            profile_id=profile_id,
            confidence=float(portfolio.get("confidence") or 0),
            category=str(portfolio.get("category") or "VERY_LOW"),
            snapshot_json=report,
        )
