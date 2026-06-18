"""Portfolio Digital Twin orchestrator."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.digital_twin.scenario_builder import (
    SCENARIO_CUSTOM,
    SCENARIO_RECOMMENDED,
    build_comparison_set,
    build_scenario,
)
from src.digital_twin.scenario_repository import ScenarioRepository
from src.digital_twin.scenario_runner import ScenarioRunner
from src.services.profile_service import ProfileContext
from src.strategy_lifecycle.qualification_engine import QualificationEngine


class TwinEngine:
    """Compare current, recommended, and policy scenarios before adoption."""

    def __init__(
        self,
        *,
        runner: ScenarioRunner | None = None,
        repository: ScenarioRepository | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections or repository is None
        self._runner = runner or ScenarioRunner()
        self._repo = repository or ScenarioRepository(owns_connection=self._owns)

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run_scenario(
        self,
        *,
        scenario_name: str,
        profile_ctx: ProfileContext,
        trades: pd.DataFrame,
        paae_report: dict[str, Any] | None = None,
        custom_allocation: dict[str, float] | None = None,
        health_score: float | None = None,
        persist: bool = True,
        created_by: str = "pdts",
        mc_fast: bool = True,
    ) -> dict[str, Any]:
        paae_report = paae_report or {}
        health = float(
            health_score
            if health_score is not None
            else (paae_report.get("health_impact") or {}).get("health_score")
            or (paae_report.get("health_report") or {}).get("health_score")
            or 50.0
        )
        scenario = build_scenario(
            scenario_name,
            profile_ctx=profile_ctx,
            paae_report=paae_report,
            custom_allocation=custom_allocation,
        )
        result = self._runner.run(
            scenario,
            trades,
            profile_ctx,
            health_score=health,
            mc_fast=mc_fast,
        )
        baseline_alloc = profile_ctx.strategy_allocations or {}
        result = self._attach_portfolio_fit_impact(
            result,
            scenario_allocation=scenario.allocation,
            baseline_allocation=baseline_alloc,
            prae_v2=paae_report,
            state_summary={"health_score": health},
            trades=trades,
        )
        if persist:
            run_id = self._repo.save_run(
                profile_id=profile_ctx.profile_id,
                scenario_name=scenario_name,
                metrics=result,
                allocation=scenario.allocation,
                created_by=created_by,
                monte_carlo=result.get("monte_carlo"),
            )
            result["run_id"] = run_id
        return result

    @staticmethod
    def _weighted_portfolio_fit(
        allocation: dict[str, float],
        fit_scores: dict[str, float],
    ) -> float:
        weights = {k: float(v) for k, v in allocation.items() if float(v) > 0}
        if not weights:
            return 0.0
        total = sum(weights.values()) or 1.0
        return round(
            sum(float(fit_scores.get(k, 0.0)) * (w / total) for k, w in weights.items()),
            1,
        )

    def _attach_portfolio_fit_impact(
        self,
        result: dict[str, Any],
        *,
        scenario_allocation: dict[str, float],
        baseline_allocation: dict[str, float],
        prae_v2: dict[str, Any] | None,
        state_summary: dict[str, Any] | None,
        trades: pd.DataFrame | None,
    ) -> dict[str, Any]:
        qualification = QualificationEngine()
        try:
            all_metrics = qualification.build_all_base_metrics(
                prae_v2=prae_v2,
                state_summary=state_summary,
                trades=trades,
            )
            fit_bundle = qualification._fit.compute_all(all_metrics, trades)
            fit_scores = {
                sid: float(data.get("portfolio_fit_score") or 0.0)
                for sid, data in fit_bundle.items()
            }
        finally:
            qualification.close()

        baseline_fit = self._weighted_portfolio_fit(baseline_allocation, fit_scores)
        scenario_fit = self._weighted_portfolio_fit(scenario_allocation, fit_scores)
        result["expected_pf"] = result.get("pf")
        result["expected_dd"] = result.get("max_dd")
        result["portfolio_fit_gain"] = round(scenario_fit - baseline_fit, 1)
        result["portfolio_fit_score"] = scenario_fit
        result["core_strategy_impact"] = self._core_strategy_impact(
            scenario_allocation,
            baseline_allocation,
            fit_scores,
        )
        return result

    @staticmethod
    def _core_strategy_impact(
        scenario_allocation: dict[str, float],
        baseline_allocation: dict[str, float],
        fit_scores: dict[str, float],
    ) -> float:
        """Weighted fit delta attributable to core-stage allocation changes."""
        core_codes = {"LSFC"}  # fallback when lifecycle repo unavailable
        try:
            from src.strategy_lifecycle.lifecycle_manager import LifecycleManager
            from src.strategy_lifecycle.lifecycle_repository import LifecycleRepository

            repo = LifecycleRepository(owns_connection=False)
            try:
                core_codes = LifecycleManager().core_strategies(repo.list_strategies())
            finally:
                repo.close()
        except Exception:
            pass

        if not core_codes:
            return 0.0

        def core_weighted(allocation: dict[str, float]) -> float:
            total = sum(allocation.values()) or 1.0
            core_total = sum(allocation.get(code, 0.0) for code in core_codes)
            if core_total <= 0:
                return 0.0
            return sum(
                float(fit_scores.get(code, 0.0)) * (allocation.get(code, 0.0) / total)
                for code in core_codes
            )

        return round(core_weighted(scenario_allocation) - core_weighted(baseline_allocation), 1)

    def compare_scenarios(
        self,
        *,
        profile_ctx: ProfileContext,
        trades: pd.DataFrame,
        paae_report: dict[str, Any] | None = None,
        include: tuple[str, ...] | None = None,
        persist: bool = False,
        mc_fast: bool = True,
    ) -> dict[str, Any]:
        paae_report = paae_report or {}
        health = float(
            (paae_report.get("health_impact") or {}).get("health_score")
            or (paae_report.get("health_report") or {}).get("health_score")
            or 50.0
        )
        scenarios = build_comparison_set(
            profile_ctx=profile_ctx,
            paae_report=paae_report,
            include=include,
        )
        baseline_alloc = profile_ctx.strategy_allocations or {}
        results: list[dict[str, Any]] = []
        for scenario in scenarios:
            row = self._runner.run(
                scenario,
                trades,
                profile_ctx,
                health_score=health,
                mc_fast=mc_fast,
                persist_mc=True,
            )
            row = self._attach_portfolio_fit_impact(
                row,
                scenario_allocation=scenario.allocation,
                baseline_allocation=baseline_alloc,
                prae_v2=paae_report,
                state_summary={"health_score": health},
                trades=trades,
            )
            if persist:
                row["run_id"] = self._repo.save_run(
                    profile_id=profile_ctx.profile_id,
                    scenario_name=scenario.name,
                    metrics=row,
                    allocation=scenario.allocation,
                    monte_carlo=row.get("monte_carlo"),
                )
            results.append(row)

        ranked = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        baseline = next((r for r in results if r["scenario"] == "baseline"), None)
        recommended = next((r for r in results if r["scenario"] == SCENARIO_RECOMMENDED), None)
        if recommended and baseline:
            recommended.setdefault(
                "portfolio_fit_gain",
                round(
                    float(recommended.get("portfolio_fit_score") or 0.0)
                    - float(baseline.get("portfolio_fit_score") or 0.0),
                    1,
                ),
            )

        return {
            "profile_id": profile_ctx.profile_id,
            "account_state": paae_report.get("account_state"),
            "scenarios": results,
            "ranking": ranked,
            "baseline": baseline,
            "recommended": recommended,
            "allocation_impact": self._allocation_impact(baseline, recommended),
        }

    @staticmethod
    def _allocation_impact(
        baseline: dict[str, Any] | None,
        recommended: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not baseline or not recommended:
            return []
        before = baseline.get("allocation") or {}
        after = recommended.get("allocation") or {}
        rows: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            b = float(before.get(key, 0.0))
            a = float(after.get(key, 0.0))
            rows.append(
                {
                    "strategy": key,
                    "before_pct": b,
                    "after_pct": a,
                    "delta_pct": round(a - b, 1),
                }
            )
        return sorted(rows, key=lambda r: abs(r["delta_pct"]), reverse=True)
