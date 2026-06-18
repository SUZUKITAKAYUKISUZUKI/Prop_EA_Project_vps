"""Multi-future strategic simulation for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v4.allocation_scenario_engine import AllocationScenarioEngine
from src.ai_governor_v4.decision_scenario_generator import DecisionScenarioGenerator
from src.ai_governor_v4.future_branch import FutureBranch
from src.ai_governor_v4.future_score_engine import FutureScoreEngine
from src.ai_governor_v4.governance_optimizer import GovernanceOptimizer
from src.ai_governor_v4.lifecycle_scenario_engine import LifecycleScenarioEngine
from src.ai_governor_v4.profile_scenario_engine import ProfileScenarioEngine
from src.ai_governor_v4.recommendation_ranker import RecommendationRanker
from src.ai_governor_v4.scenario_comparator import ScenarioComparator
from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG


class StrategicSimulator:
    """Evaluate multiple governance futures — never places trades."""

    def __init__(self, config: StrategicConfig | None = None) -> None:
        self._config = config or DEFAULT_STRATEGIC_CONFIG
        self._generator = DecisionScenarioGenerator()
        self._profile = ProfileScenarioEngine(self._config)
        self._allocation = AllocationScenarioEngine(self._config)
        self._lifecycle = LifecycleScenarioEngine(self._config)
        self._scorer = FutureScoreEngine(self._config)
        self._ranker = RecommendationRanker()
        self._comparator = ScenarioComparator()
        self._optimizer = GovernanceOptimizer(self._ranker)

    def simulate(
        self,
        context: GovernorContext,
        *,
        age_v3_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        specs = self._generator.generate(context, age_v3_report=age_v3_report)
        branches: list[FutureBranch] = []
        baseline_metrics: dict[str, dict[str, float]] | None = None

        for spec in specs:
            modified = self._generator.apply_modification(context, spec.get("modification") or {})
            metrics = self._evaluate_branch(modified, spec)
            if spec.get("action_type") == "DO_NOTHING":
                baseline_metrics = metrics

            score_result = self._scorer.score_branch(metrics, baseline_metrics=baseline_metrics)
            branch = FutureBranch(
                branch_id=str(spec.get("branch_id") or ""),
                action_type=str(spec.get("action_type") or ""),
                action_label=str(spec.get("action_label") or ""),
                description=str(spec.get("description") or ""),
                metrics_by_horizon=metrics,
                strategic_score=float(score_result.get("strategic_score") or 0),
                expected_benefit=float(score_result.get("expected_benefit") or 0),
                expected_risk=float(score_result.get("expected_risk") or 0),
                modification=dict(spec.get("modification") or {}),
            )
            branches.append(branch)

        if baseline_metrics is None and branches:
            baseline_metrics = branches[0].metrics_by_horizon

        for branch in branches:
            if branch.action_type != "DO_NOTHING":
                score_result = self._scorer.score_branch(branch.metrics_by_horizon, baseline_metrics=baseline_metrics)
                branch.strategic_score = float(score_result.get("strategic_score") or 0)
                branch.expected_benefit = float(score_result.get("expected_benefit") or 0)
                branch.expected_risk = float(score_result.get("expected_risk") or 0)

        ranked = self._ranker.rank_branches(branches)
        baseline_branch = next((b for b in ranked if b.action_type == "DO_NOTHING"), None)
        comparison = self._comparator.compare(ranked, baseline=baseline_branch)
        optimization = self._optimizer.select_best(ranked)
        rankings = self._ranker.build_rankings(ranked)

        return {
            "branches": ranked,
            "comparison": comparison,
            "optimization": optimization,
            "rankings": rankings,
            "baseline_metrics": baseline_metrics or {},
        }

    def _evaluate_branch(self, context: GovernorContext, spec: dict[str, Any]) -> dict[str, dict[str, float]]:
        action_type = str(spec.get("action_type") or "")
        modification = dict(spec.get("modification") or {})

        if action_type in {"REDUCE_ALLOCATION", "INCREASE_ALLOCATION"}:
            return self._allocation.evaluate(context, action_type=action_type, modification=modification)
        if action_type in {"PROMOTE_STRATEGY", "DEMOTE_STRATEGY"}:
            return self._lifecycle.evaluate(context, action_type=action_type, modification=modification)
        return self._profile.evaluate(context, action_type=action_type)
