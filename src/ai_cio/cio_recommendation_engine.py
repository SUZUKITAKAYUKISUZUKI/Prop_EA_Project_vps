"""Executive recommendation synthesis for AI CIO."""
from __future__ import annotations

from typing import Any

from src.ai_cio.cio_allocation_engine import CioAllocationEngine
from src.ai_cio.cio_priority_engine import CioPriorityEngine
from src.ai_cio.cio_profile_engine import CioProfileEngine
from src.ai_cio.cio_strategy_engine import CioStrategyEngine


class CioRecommendationEngine:
    def __init__(
        self,
        *,
        strategy_engine: CioStrategyEngine | None = None,
        allocation_engine: CioAllocationEngine | None = None,
        profile_engine: CioProfileEngine | None = None,
        priority_engine: CioPriorityEngine | None = None,
    ) -> None:
        self._strategy = strategy_engine or CioStrategyEngine()
        self._allocation = allocation_engine or CioAllocationEngine()
        self._profile = profile_engine or CioProfileEngine()
        self._priority = priority_engine or CioPriorityEngine()

    def evaluate(
        self,
        context: dict[str, Any],
        *,
        opinion: str,
        recommended_profile: str,
    ) -> dict[str, Any]:
        strategy_recs = self._strategy.evaluate(context)
        allocation_recs = self._allocation.evaluate(context)
        profile_recs = self._profile.evaluate(context, recommended_profile=recommended_profile)
        risk_recs = self._priority.build_risk_recommendations(context)
        portfolio_recs = self._priority.build_portfolio_recommendations(context, opinion=opinion)

        recommendations = self._priority.evaluate(
            context=context,
            strategy_recs=strategy_recs,
            allocation_recs=allocation_recs,
            profile_recs=profile_recs,
            risk_recs=risk_recs,
            portfolio_recs=portfolio_recs,
        )

        return {
            "recommendations": recommendations,
            "recommended_actions": [r.get("description") for r in recommendations[:5]],
            "top_recommendation": recommendations[0] if recommendations else None,
        }
