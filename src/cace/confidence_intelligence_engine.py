"""CACE v1.5 confidence intelligence orchestration."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_breakdown_engine import ConfidenceBreakdownEngine
from src.cace.confidence_cache import ConfidenceCache
from src.cace.confidence_history_engine import ConfidenceHistoryEngine
from src.cace.confidence_intelligence_report import ConfidenceIntelligenceReport
from src.cace.confidence_regime_engine import ConfidenceRegimeEngine
from src.cace.confidence_trend_engine import ConfidenceTrendEngine
from src.cace.confidence_v15_config import (
    CACHE_V15_HISTORY,
    CACHE_V15_PORTFOLIO,
    CACHE_V15_TREND,
)


class ConfidenceIntelligenceEngine:
    """Aggregates breakdown, trend, regime, and history — evaluation only."""

    def __init__(
        self,
        *,
        breakdown: ConfidenceBreakdownEngine | None = None,
        trend: ConfidenceTrendEngine | None = None,
        history: ConfidenceHistoryEngine | None = None,
        regime: ConfidenceRegimeEngine | None = None,
        reporter: ConfidenceIntelligenceReport | None = None,
        cache: ConfidenceCache | None = None,
    ) -> None:
        self._breakdown = breakdown or ConfidenceBreakdownEngine()
        self._trend = trend or ConfidenceTrendEngine()
        self._history = history or ConfidenceHistoryEngine()
        self._regime = regime or ConfidenceRegimeEngine()
        self._reporter = reporter or ConfidenceIntelligenceReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def run(
        self,
        *,
        profile_id: str,
        cace_v1_report: dict[str, Any],
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_V15_PORTFOLIO.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        portfolio = cace_v1_report.get("portfolio_confidence") or {}
        allocation = cace_v1_report.get("allocation_confidence") or {}
        factors = allocation.get("factors") or portfolio.get("factors") or {}

        breakdown = self._breakdown.build(factors)
        raw_confidence = float(cace_v1_report.get("confidence") or portfolio.get("confidence") or 0)

        hist_rows = self._history.list_history(profile_id=profile_id, limit=90)
        trend = self._trend.analyze(hist_rows, current_confidence=raw_confidence)
        regime = self._regime.evaluate(
            profile_id=profile_id,
            raw_confidence=raw_confidence,
            persist=persist,
            use_cache=use_cache,
        )
        stability = self._history.stability_score(hist_rows)
        timeline = self._history.build_timeline(hist_rows)

        report = self._reporter.build(
            profile_id=profile_id,
            cace_v1_report=cace_v1_report,
            breakdown=breakdown,
            trend=trend,
            regime=regime,
            history=timeline,
            stability_score=stability,
            top_drivers=self._breakdown.top_drivers(breakdown),
            top_risks=self._breakdown.top_risks(breakdown),
        )

        if persist:
            self._history.record(
                profile_id=profile_id,
                confidence=float(report.get("confidence") or 0),
                category=str(report.get("confidence_category") or "VERY_LOW"),
                trend=str(report.get("trend") or "FLAT"),
                trend_strength=float(report.get("trend_strength") or 0),
                snapshot_json=report,
            )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_V15_TREND.format(profile_id=profile_id), trend)
            self._cache.set(CACHE_V15_HISTORY.format(profile_id=profile_id), timeline)

        self._last_report = report
        return report

    def enrich_age_v4_report(self, age_v4_report: dict[str, Any], intelligence: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(age_v4_report)
        enriched["confidence"] = intelligence.get("confidence")
        enriched["raw_confidence"] = intelligence.get("raw_confidence")
        enriched["regime_modifier"] = intelligence.get("regime_modifier")
        enriched["confidence_category"] = intelligence.get("confidence_category")
        enriched["trend"] = intelligence.get("trend")
        enriched["trend_strength"] = intelligence.get("trend_strength")
        enriched["trend_direction"] = intelligence.get("trend_direction")
        enriched["regime"] = intelligence.get("regime")
        enriched["breakdown"] = intelligence.get("breakdown")
        enriched["cace_v15"] = intelligence
        return enriched

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}
