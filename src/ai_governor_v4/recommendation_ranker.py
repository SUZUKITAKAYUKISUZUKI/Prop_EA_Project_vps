"""Rank strategic branches for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor_v4.future_branch import FutureBranch
from src.ai_governor_v4.strategic_config import RANK_THRESHOLDS


class RecommendationRanker:
    def rank_category(self, strategic_score: float) -> str:
        for threshold, label in RANK_THRESHOLDS:
            if strategic_score >= threshold:
                return label
        return "REJECT"

    def rank_branches(self, branches: list[FutureBranch]) -> list[FutureBranch]:
        ranked = sorted(branches, key=lambda b: b.strategic_score, reverse=True)
        for idx, branch in enumerate(ranked, start=1):
            branch.rank_category = self.rank_category(branch.strategic_score)
            branch.confidence = self._confidence(branch, rank_position=idx, total=len(ranked))
        return ranked

    def build_rankings(self, branches: list[FutureBranch]) -> dict[str, Any]:
        ranked = self.rank_branches(branches)
        return {
            "rankings": [
                {
                    "rank": idx,
                    "branch_id": b.branch_id,
                    "action_label": b.action_label,
                    "action_type": b.action_type,
                    "strategic_score": b.strategic_score,
                    "rank_category": b.rank_category,
                    "confidence": b.confidence,
                    "expected_benefit": b.expected_benefit,
                    "expected_risk": b.expected_risk,
                }
                for idx, b in enumerate(ranked, start=1)
            ],
            "strongly_recommended": [
                b.action_label for b in ranked if b.rank_category == "STRONGLY_RECOMMENDED"
            ],
            "recommended": [b.action_label for b in ranked if b.rank_category == "RECOMMENDED"],
            "rejected": [b.action_label for b in ranked if b.rank_category == "REJECT"],
        }

    def _confidence(self, branch: FutureBranch, *, rank_position: int, total: int) -> float:
        base = min(100.0, branch.strategic_score * 0.85 + 10.0)
        if branch.action_type == "DO_NOTHING":
            base = max(50.0, base - 5.0)
        rank_bonus = max(0.0, (total - rank_position) / max(1, total) * 8.0)
        benefit_bonus = min(10.0, max(0.0, branch.expected_benefit))
        risk_penalty = min(15.0, branch.expected_risk * 0.1)
        return round(min(100.0, max(0.0, base + rank_bonus + benefit_bonus - risk_penalty)), 1)
