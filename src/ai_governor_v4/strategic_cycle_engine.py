"""Core strategic cycle orchestration for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v4.future_tree_builder import FutureTreeBuilder
from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG
from src.ai_governor_v4.strategic_report import StrategicReport
from src.ai_governor_v4.strategic_repository import StrategicRepository
from src.ai_governor_v4.strategic_simulator import StrategicSimulator
from src.repositories.cache_repository import CacheRepository


class StrategicCycleEngine:
    def __init__(
        self,
        *,
        config: StrategicConfig | None = None,
        simulator: StrategicSimulator | None = None,
        reporter: StrategicReport | None = None,
        tree_builder: FutureTreeBuilder | None = None,
        repo: StrategicRepository | None = None,
        cache: CacheRepository | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._config = config or DEFAULT_STRATEGIC_CONFIG
        self._owns = owns_connections or repo is None
        self._simulator = simulator or StrategicSimulator(self._config)
        self._reporter = reporter or StrategicReport()
        self._tree = tree_builder or FutureTreeBuilder(self._config)
        self._repo = repo or StrategicRepository(owns_connection=self._owns)
        self._cache = cache or CacheRepository(owns_connection=False)
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        context: GovernorContext,
        *,
        age_v3_report: dict[str, Any],
        persist: bool = True,
    ) -> dict[str, Any]:
        cache_key = f"{self._config.cache_key_prefix}:{context.profile_id}"
        if self._config.use_cache and not persist:
            cached = self._cache.get(cache_key)
            if cached:
                self._last_report = cached
                return cached

        simulation = self._simulator.simulate(context, age_v3_report=age_v3_report)
        branches = simulation.get("branches") or []
        future_tree = self._tree.build(context, branches)
        report = self._reporter.build(
            context=context,
            simulation=simulation,
            future_tree=future_tree,
            age_v3_report=age_v3_report,
        )
        report["age_v3"] = age_v3_report

        if persist:
            scenario_id = self._repo.save_scenario(
                profile_id=context.profile_id,
                horizon_label=self._config.horizons_label(),
                scenario_json={
                    "future_tree": future_tree,
                    "comparison": simulation.get("comparison"),
                    "optimization": simulation.get("optimization"),
                },
                confidence=float(report.get("confidence") or 0),
            )
            self._repo.save_branches(scenario_id, [b.to_dict() for b in branches])
            opt = simulation.get("optimization") or {}
            self._repo.save_rankings(
                scenario_id,
                rankings_json=simulation.get("rankings") or {},
                best_branch_id=str(opt.get("branch_id") or ""),
                best_action=str(opt.get("recommended_action") or ""),
                confidence=float(opt.get("confidence") or 0),
            )

        if self._config.use_cache:
            self._cache.set(cache_key, report)

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}
