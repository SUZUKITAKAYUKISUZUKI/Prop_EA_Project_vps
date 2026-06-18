"""AGE v4 Strategic Governor — top-level engine."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.engine import PredictiveGovernorEngine
from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG
from src.ai_governor_v4.strategic_cycle_engine import StrategicCycleEngine
from src.ai_governor_v4.strategic_repository import StrategicRepository
from src.cace.confidence_engine import ConfidenceEngine
from src.cace.confidence_intelligence_engine import ConfidenceIntelligenceEngine


class StrategicGovernorEngine:
    """AGE v4 extends AGE v3 with multi-future strategic simulation — never places trades."""

    def __init__(
        self,
        *,
        age_v3: PredictiveGovernorEngine | None = None,
        strategic: StrategicCycleEngine | None = None,
        repo: StrategicRepository | None = None,
        cace: ConfidenceEngine | None = None,
        cace_v15: ConfidenceIntelligenceEngine | None = None,
        config: StrategicConfig | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._config = config or DEFAULT_STRATEGIC_CONFIG
        self._age_v3 = age_v3 or PredictiveGovernorEngine(owns_connections=owns_connections)
        self._strategic = strategic or StrategicCycleEngine(config=self._config, owns_connections=owns_connections)
        self._repo = repo or StrategicRepository(owns_connection=False)
        self._cace = cace or ConfidenceEngine(owns_connections=False)
        self._cace_v15 = cace_v15 or ConfidenceIntelligenceEngine()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        self._strategic.close()
        if self._owns:
            self._age_v3.close()

    def run_strategic_cycle(
        self,
        context: GovernorContext,
        *,
        age_v2_report: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        v3_report = self._age_v3.run_predictive_cycle(
            context,
            age_v2_report=age_v2_report,
            persist=False,
        )
        report = self._strategic.run(context, age_v3_report=v3_report, persist=persist)
        report["age_v3"] = v3_report
        if age_v2_report:
            report["age_v2"] = age_v2_report
        elif v3_report.get("age_v2"):
            report["age_v2"] = v3_report["age_v2"]
        report = self._enrich_with_cace(context, report, persist=persist)
        self._last_report = report
        return report

    def _enrich_with_cace(
        self,
        context: GovernorContext,
        report: dict[str, Any],
        *,
        persist: bool,
    ) -> dict[str, Any]:
        cace_report = self._cace.run(
            profile_id=context.profile_id,
            paae=context.paae,
            pdts=context.pdts,
            prae_v2=context.prae_v2,
            state_analytics=context.state_analytics,
            slm=context.slm,
            age_v4=report,
            current_weights=context.current_allocation,
            persist=persist,
            use_cache=not persist,
        )
        enriched = self._cace.enrich_age_v4_report(report, cace_report)
        intelligence = self._cace_v15.run(
            profile_id=context.profile_id,
            cace_v1_report=cace_report,
            persist=persist,
            use_cache=not persist,
        )
        enriched = self._cace_v15.enrich_age_v4_report(enriched, intelligence)
        enriched["cace_report"] = cace_report
        enriched["cace_v15_report"] = intelligence
        return enriched

    def get_strategic_report(self) -> dict[str, Any]:
        return self._last_report or {}

    def get_future_scenarios(self) -> list[dict[str, Any]]:
        report = self.get_strategic_report()
        return report.get("future_scenarios") or []

    def get_future_rankings(self) -> dict[str, Any]:
        report = self.get_strategic_report()
        return report.get("future_rankings") or {}

    def get_best_future(self) -> dict[str, Any]:
        report = self.get_strategic_report()
        return {
            "recommended_action": report.get("recommended_action"),
            "confidence": report.get("confidence"),
            "confidence_category": report.get("confidence_category"),
            "strategic_score": report.get("strategic_score"),
            "future_score": report.get("future_score") or report.get("strategic_score"),
            "strategic_confidence": report.get("strategic_confidence"),
            "rank_category": report.get("rank_category"),
            "expected_benefit": report.get("expected_benefit"),
            "expected_risk": report.get("expected_risk"),
            "rationale": report.get("rationale"),
            "best_future_metrics": report.get("best_future_metrics"),
        }

    def get_scenario_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._repo.list_scenarios(limit=limit)

    def get_ranking_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._repo.list_rankings(limit=limit)
