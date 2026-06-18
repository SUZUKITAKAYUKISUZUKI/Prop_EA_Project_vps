"""Portfolio-level confidence aggregation for CACE."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_calculator import ConfidenceCalculator
from src.cace.confidence_explainability import ConfidenceExplainability
from src.cace.confidence_factors import ConfidenceFactors


class PortfolioConfidenceEngine:
    def __init__(self) -> None:
        self._calculator = ConfidenceCalculator()
        self._explain = ConfidenceExplainability()

    def evaluate(
        self,
        *,
        profile_id: str,
        factors: ConfidenceFactors,
        allocation_confidence: dict[str, Any],
        strategy_confidences: list[dict[str, Any]],
        age_v4: dict[str, Any],
        paae: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = self._calculator.compute(factors)
        alloc_conf = float(allocation_confidence.get("confidence") or base)
        strat_scores = [float(s.get("confidence") or 0) for s in strategy_confidences]
        strat_avg = sum(strat_scores) / len(strat_scores) if strat_scores else alloc_conf

        age_bonus = 0.0
        if age_v4.get("rank_category") in {"STRONGLY_RECOMMENDED", "RECOMMENDED"}:
            age_bonus = 5.0
        governance_conf = float(age_v4.get("strategic_confidence") or age_v4.get("confidence") or 0)
        if governance_conf >= 80:
            age_bonus += 3.0

        confidence = round(min(100.0, alloc_conf * 0.55 + strat_avg * 0.30 + base * 0.15 + age_bonus), 1)
        category = self._calculator.category(confidence)

        return {
            "profile_id": profile_id,
            "confidence": confidence,
            "category": category,
            "allocation_confidence": alloc_conf,
            "strategy_confidence_avg": round(strat_avg, 1),
            "factors": factors.to_dict(),
            "top_risks": self._explain.top_risks(factors, age_v4=age_v4),
            "top_opportunities": self._explain.top_opportunities(factors, paae=paae or {}),
            "reason": self._explain.portfolio_reasons(
                factors,
                allocation_confidence=alloc_conf,
                strategy_confidences=strat_scores,
            ),
        }
