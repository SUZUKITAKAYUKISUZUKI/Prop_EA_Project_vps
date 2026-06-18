"""Strategic governor report builder for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v4.future_branch import FutureBranch
from src.ai_governor_v4.future_tree_builder import FutureTreeBuilder


class StrategicReport:
    def build(
        self,
        *,
        context: GovernorContext,
        simulation: dict[str, Any],
        future_tree: dict[str, Any],
        age_v3_report: dict[str, Any],
    ) -> dict[str, Any]:
        branches: list[FutureBranch] = simulation.get("branches") or []
        optimization = simulation.get("optimization") or {}
        comparison = simulation.get("comparison") or {}
        rankings = simulation.get("rankings") or {}

        primary_horizon = "90d"
        best_branch = next(
            (b for b in branches if b.branch_id == optimization.get("branch_id")),
            branches[0] if branches else None,
        )
        best_metrics = (best_branch.metrics_by_horizon.get(primary_horizon) or {}) if best_branch else {}

        return {
            "profile_id": context.profile_id,
            "current_state": context.current_state,
            "current_profile": context.current_profile,
            "health_score": context.health_score,
            "dd_pct": context.dd_pct,
            "forecast_horizons": future_tree.get("forecast_horizons"),
            "future_tree": future_tree,
            "future_scenarios": [b.to_dict() for b in branches],
            "scenario_comparison": comparison,
            "future_rankings": rankings,
            "recommended_action": optimization.get("recommended_action"),
            "action_type": optimization.get("action_type"),
            "confidence": optimization.get("confidence"),
            "strategic_score": optimization.get("strategic_score"),
            "rank_category": optimization.get("rank_category"),
            "expected_benefit": optimization.get("expected_benefit"),
            "expected_risk": optimization.get("expected_risk"),
            "rationale": optimization.get("rationale"),
            "future_health": self._extract_metric(branches, "health_score"),
            "future_dd": self._extract_metric(branches, "expected_dd"),
            "future_pf": self._extract_metric(branches, "expected_pf"),
            "best_future_metrics": best_metrics,
            "age_v3_summary": {
                "confidence": age_v3_report.get("confidence"),
                "health_forecast": age_v3_report.get("health_forecast"),
                "predictive_recommendations": age_v3_report.get("predictive_recommendations"),
            },
        }

    def _extract_metric(self, branches: list[FutureBranch], key: str) -> dict[str, Any]:
        rows = {}
        for branch in branches:
            metrics_90 = branch.metrics_by_horizon.get("90d") or {}
            rows[branch.action_label] = metrics_90.get(key)
        return rows


class StrategicReportBuilder(StrategicReport):
    """Alias for report builder used by engine."""

    def build_tree(self, context: GovernorContext, branches: list[FutureBranch]) -> dict[str, Any]:
        return FutureTreeBuilder().build(context, branches)
