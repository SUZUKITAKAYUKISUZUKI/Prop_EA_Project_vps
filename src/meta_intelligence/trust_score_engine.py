"""Trust score computation for Portfolio OS modules."""
from __future__ import annotations

from src.meta_intelligence.config import TRUST_WEIGHTS, trust_category


class TrustScoreEngine:
    def evaluate(self, components: dict[str, dict[str, float]]) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        for module, scores in components.items():
            trust_score = round(
                TRUST_WEIGHTS["historical_accuracy"] * scores["historical_accuracy"]
                + TRUST_WEIGHTS["calibration"] * scores["calibration"]
                + TRUST_WEIGHTS["stability"] * scores["stability"]
                + TRUST_WEIGHTS["consensus"] * scores["consensus"]
                + TRUST_WEIGHTS["predictive_reliability"] * scores["predictive_reliability"],
                2,
            )
            results[module] = {
                "module": module,
                "trust_score": trust_score,
                "category": trust_category(trust_score),
                "components": scores,
            }
        return results
