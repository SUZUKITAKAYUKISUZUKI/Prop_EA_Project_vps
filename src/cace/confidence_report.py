"""CACE report builder."""
from __future__ import annotations

from typing import Any


class ConfidenceReport:
    def build(
        self,
        *,
        profile_id: str,
        allocation_confidence: dict[str, Any],
        strategy_confidences: list[dict[str, Any]],
        portfolio_confidence: dict[str, Any],
        confidence_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trend = self._confidence_trend(confidence_history)
        return {
            "profile_id": profile_id,
            "allocation_confidence": allocation_confidence,
            "strategy_confidence": strategy_confidences,
            "portfolio_confidence": portfolio_confidence,
            "confidence": portfolio_confidence.get("confidence"),
            "confidence_category": portfolio_confidence.get("category"),
            "confidence_history": confidence_history[:10],
            "confidence_trend": trend,
            "top_risks": portfolio_confidence.get("top_risks") or [],
            "top_opportunities": portfolio_confidence.get("top_opportunities") or [],
        }

    def _confidence_trend(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if len(history) < 2:
            return {"direction": "stable", "delta": 0.0}
        latest = float(history[0].get("confidence") or 0)
        previous = float(history[1].get("confidence") or 0)
        delta = round(latest - previous, 1)
        if delta > 2:
            direction = "up"
        elif delta < -2:
            direction = "down"
        else:
            direction = "stable"
        return {"direction": direction, "delta": delta, "latest": latest, "previous": previous}
