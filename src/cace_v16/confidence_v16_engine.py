"""CACE v1.6 combined intelligence orchestration."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.cace_v16.confidence_consensus_engine import ConfidenceConsensusEngine
from src.cace_v16.confidence_decay_engine import ConfidenceDecayEngine
from src.cace_v16.confidence_v16_models import (
    CACHE_V16_CONSENSUS,
    CACHE_V16_DECAY,
    CACHE_V16_INTELLIGENCE,
)
from src.cace_v16.confidence_v16_report import ConfidenceV16Report
from src.cace_v16.confidence_v16_repository import ConfidenceV16Repository


class ConfidenceV16Engine:
    """Aggregates CACE v1.5, decay, and consensus — evaluation only."""

    def __init__(
        self,
        *,
        decay_engine: ConfidenceDecayEngine | None = None,
        consensus_engine: ConfidenceConsensusEngine | None = None,
        repo: ConfidenceV16Repository | None = None,
        reporter: ConfidenceV16Report | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._decay = decay_engine or ConfidenceDecayEngine()
        self._consensus = consensus_engine or ConfidenceConsensusEngine()
        self._repo = repo or ConfidenceV16Repository(owns_connection=self._owns)
        self._reporter = reporter or ConfidenceV16Report()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        cace_v15_report: dict[str, Any],
        upstream: dict[str, Any] | None = None,
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_V16_INTELLIGENCE.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        upstream = upstream or {}
        decay = self._decay.evaluate(cace_v15_report)
        consensus = self._consensus.evaluate(
            paae=upstream.get("paae"),
            pdts=upstream.get("pdts"),
            slm=upstream.get("slm"),
            age_v4=upstream.get("age_v4"),
            prae_v2=upstream.get("prae_v2"),
            state_analytics=upstream.get("state_analytics"),
            explicit_recommendations=upstream.get("participant_votes") or upstream.get("explicit_recommendations"),
        )
        report = self._reporter.build(
            profile_id=profile_id,
            cace_v15_report=cace_v15_report,
            decay=decay,
            consensus=consensus,
        )

        if persist:
            self._repo.save_decay(
                profile_id=profile_id,
                durability_score=float(decay.get("durability_score") or 0),
                durability_category=str(decay.get("durability_category") or "VERY_LOW"),
                half_life=int(decay.get("half_life") or 60),
                forecast_json=decay.get("forecast") or {},
            )
            self._repo.save_consensus(
                profile_id=profile_id,
                recommended_action=str(consensus.get("recommended_action") or "NO_ACTION"),
                consensus_score=float(consensus.get("consensus_score") or 25),
                consensus_category=str(consensus.get("consensus_category") or "VERY_LOW"),
                agreement_ratio=float(consensus.get("agreement_ratio") or 0),
                agree_count=int(consensus.get("agree_count") or 0),
                total_modules=int(consensus.get("total_modules") or 0),
            )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_V16_DECAY.format(profile_id=profile_id), decay)
            self._cache.set(CACHE_V16_CONSENSUS.format(profile_id=profile_id), consensus)

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}

    def enrich_v15_report(self, v15_report: dict[str, Any], v16_report: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(v15_report)
        enriched.update(
            {
                "durability_score": v16_report.get("durability_score"),
                "durability_category": v16_report.get("durability_category"),
                "half_life": v16_report.get("half_life"),
                "forecast": v16_report.get("forecast"),
                "consensus_score": v16_report.get("consensus_score"),
                "consensus_category": v16_report.get("consensus_category"),
                "agreement_ratio": v16_report.get("agreement_ratio"),
                "recommended_action": v16_report.get("recommended_action"),
                "cace_v16": v16_report,
            }
        )
        return enriched
