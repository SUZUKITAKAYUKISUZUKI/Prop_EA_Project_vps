"""Meta Intelligence Engine v1 orchestration — audit only."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.meta_intelligence.config import (
    CACHE_MIE_DRIFT,
    CACHE_MIE_IMPROVEMENTS,
    CACHE_MIE_INTELLIGENCE,
    CACHE_MIE_RANKING,
    CACHE_MIE_TRUST,
)
from src.meta_intelligence.module_drift_engine import ModuleDriftEngine
from src.meta_intelligence.module_ranking_engine import ModuleRankingEngine
from src.meta_intelligence.module_score_engine import ModuleScoreEngine
from src.meta_intelligence.recommendation_engine import RecommendationEngine
from src.meta_intelligence.report import MetaIntelligenceReport
from src.meta_intelligence.repository import MetaIntelligenceRepository
from src.meta_intelligence.self_improvement_engine import SelfImprovementEngine
from src.meta_intelligence.trust_score_engine import TrustScoreEngine


class MetaIntelligenceEngine:
    """Portfolio OS self-audit layer — no trades, no allocation changes."""

    def __init__(
        self,
        *,
        score_engine: ModuleScoreEngine | None = None,
        trust_engine: TrustScoreEngine | None = None,
        drift_engine: ModuleDriftEngine | None = None,
        ranking_engine: ModuleRankingEngine | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        improvement_engine: SelfImprovementEngine | None = None,
        repo: MetaIntelligenceRepository | None = None,
        reporter: MetaIntelligenceReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._scores = score_engine or ModuleScoreEngine()
        self._trust = trust_engine or TrustScoreEngine()
        self._drift = drift_engine or ModuleDriftEngine()
        self._ranking = ranking_engine or ModuleRankingEngine()
        self._recommendation = recommendation_engine or RecommendationEngine()
        self._improvement = improvement_engine or SelfImprovementEngine()
        self._repo = repo or MetaIntelligenceRepository(owns_connection=self._owns)
        self._reporter = reporter or MetaIntelligenceReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        cace_v16_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        upstream: dict[str, Any] | None = None,
        trust_history: list[dict[str, Any]] | None = None,
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_MIE_INTELLIGENCE.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        upstream = upstream or {}
        history = trust_history or self._repo.load_trust_history(profile_id=profile_id)
        components = self._scores.evaluate(
            cace_v16_report=cace_v16_report,
            cace_v17_report=cace_v17_report,
            upstream=upstream,
        )
        trust_scores = self._trust.evaluate(components)
        rankings = self._ranking.evaluate(trust_scores)
        strongest_weakest = self._ranking.strongest_weakest(rankings)
        drift = self._drift.evaluate(current_trust=trust_scores, trust_history=history)
        recommendations = self._recommendation.evaluate(
            trust_scores=trust_scores,
            rankings=rankings,
            cace_v16_report=cace_v16_report,
        )
        improvements = self._improvement.evaluate(
            trust_scores=trust_scores,
            drift=drift,
            cace_v17_report=cace_v17_report,
        )

        report = self._reporter.build(
            profile_id=profile_id,
            trust_scores=trust_scores,
            rankings=rankings,
            drift=drift,
            recommendations=recommendations,
            improvements=improvements,
            strongest_weakest=strongest_weakest,
            cace_v17_report=cace_v17_report,
        )

        if persist:
            self._repo.save_trust_scores(profile_id=profile_id, trust_scores=trust_scores)
            self._repo.save_rankings(profile_id=profile_id, rankings=rankings)
            if drift.get("drift_alerts"):
                self._repo.save_drift_alerts(profile_id=profile_id, alerts=drift["drift_alerts"])
            self._repo.save_improvement_notes(profile_id=profile_id, notes=improvements)

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_MIE_TRUST.format(profile_id=profile_id), trust_scores)
            self._cache.set(CACHE_MIE_RANKING.format(profile_id=profile_id), rankings)
            self._cache.set(CACHE_MIE_DRIFT.format(profile_id=profile_id), drift)
            self._cache.set(CACHE_MIE_IMPROVEMENTS.format(profile_id=profile_id), improvements)

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}
