"""Select optimal governance branch for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor_v4.future_branch import FutureBranch
from src.ai_governor_v4.recommendation_ranker import RecommendationRanker


class GovernanceOptimizer:
    def __init__(self, ranker: RecommendationRanker | None = None) -> None:
        self._ranker = ranker or RecommendationRanker()

    def select_best(
        self,
        branches: list[FutureBranch],
        *,
        min_acceptable_score: float = 50.0,
    ) -> dict[str, Any]:
        if not branches:
            return {
                "recommended_action": "NO_ACTION",
                "confidence": 0.0,
                "branch_id": None,
                "strategic_score": 0.0,
                "rank_category": "REJECT",
                "rationale": "No strategic branches available.",
            }

        ranked = self._ranker.rank_branches(branches)
        baseline = next((b for b in ranked if b.action_type == "DO_NOTHING"), ranked[-1])

        actionable = [b for b in ranked if b.action_type != "DO_NOTHING" and b.strategic_score >= min_acceptable_score]
        if actionable:
            best = actionable[0]
        else:
            best = baseline

        if best.action_type == "DO_NOTHING" or best.strategic_score < baseline.strategic_score + 2.0:
            best = baseline
            rationale = "Baseline (do nothing) remains optimal — no action materially improves strategic score."
        else:
            rationale = (
                f"{best.action_label} scores {best.strategic_score:.1f} vs baseline "
                f"{baseline.strategic_score:.1f} (+{best.expected_benefit:.1f} expected benefit)."
            )

        return {
            "recommended_action": best.action_label,
            "action_type": best.action_type,
            "confidence": best.confidence,
            "branch_id": best.branch_id,
            "strategic_score": best.strategic_score,
            "rank_category": best.rank_category,
            "expected_benefit": best.expected_benefit,
            "expected_risk": best.expected_risk,
            "baseline_score": baseline.strategic_score,
            "baseline_action": baseline.action_label,
            "rationale": rationale,
        }
