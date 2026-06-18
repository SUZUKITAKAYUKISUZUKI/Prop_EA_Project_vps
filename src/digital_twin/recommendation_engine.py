"""Composite recommendation scoring for PDTS scenarios."""
from __future__ import annotations

from typing import Any

RECOMMENDATION_CATEGORIES = (
    (90.0, "STRONGLY_RECOMMENDED"),
    (75.0, "RECOMMENDED"),
    (60.0, "ACCEPTABLE"),
    (40.0, "HIGH_RISK"),
    (0.0, "REJECT"),
)

WEIGHTS = {
    "pass_rate": 0.35,
    "health_score": 0.20,
    "pf": 0.15,
    "recovery_factor": 0.15,
    "sharpe": 0.15,
}


def _norm(value: float, *, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(100.0, float(value) / cap * 100.0))


class RecommendationEngine:
    def score(self, metrics: dict[str, Any]) -> float:
        pass_rate = _norm(metrics.get("pass_rate", 0.0), cap=100.0)
        health = _norm(metrics.get("health_score", 0.0), cap=100.0)
        pf = _norm(min(float(metrics.get("pf", 0.0)), 5.0), cap=5.0)
        recovery = _norm(min(float(metrics.get("recovery_factor", 0.0)), 10.0), cap=10.0)
        sharpe = _norm(min(float(metrics.get("sharpe", 0.0)), 3.0), cap=3.0)

        raw = (
            WEIGHTS["pass_rate"] * pass_rate
            + WEIGHTS["health_score"] * health
            + WEIGHTS["pf"] * pf
            + WEIGHTS["recovery_factor"] * recovery
            + WEIGHTS["sharpe"] * sharpe
        )
        return round(raw, 1)

    def categorize(self, score: float) -> str:
        for threshold, label in RECOMMENDATION_CATEGORIES:
            if score >= threshold:
                return label
        return "REJECT"

    def evaluate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        score = self.score(metrics)
        return {
            "score": score,
            "recommendation": self.categorize(score),
        }
