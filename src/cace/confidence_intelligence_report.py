"""CACE v1.5 intelligence report builder."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_calculator import ConfidenceCalculator


class ConfidenceIntelligenceReport:
    def build(
        self,
        *,
        profile_id: str,
        cace_v1_report: dict[str, Any],
        breakdown: dict[str, float],
        trend: dict[str, Any],
        regime: dict[str, Any],
        history: list[dict[str, Any]],
        stability_score: float,
        top_drivers: list[str],
        top_risks: list[str],
    ) -> dict[str, Any]:
        raw = float(cace_v1_report.get("confidence") or 0)
        adjusted = float(regime.get("adjusted_confidence") or raw)
        category = ConfidenceCalculator().category(adjusted)

        return {
            "profile_id": profile_id,
            "confidence": adjusted,
            "raw_confidence": round(raw, 1),
            "regime_modifier": float(regime.get("confidence_modifier") or 0),
            "adjusted_confidence": adjusted,
            "confidence_category": category,
            "breakdown": breakdown,
            "trend": trend.get("trend"),
            "trend_direction": trend.get("trend_direction"),
            "trend_strength": trend.get("trend_strength"),
            "trend_category": trend.get("trend_category"),
            "trend_windows": trend.get("windows"),
            "confidence_evolution": trend.get("evolution"),
            "regime": regime.get("regime"),
            "regime_metrics": regime.get("metrics"),
            "regime_rationale": regime.get("rationale"),
            "regime_appropriate": regime.get("regime_appropriate"),
            "confidence_history": history[:20],
            "confidence_stability": stability_score,
            "top_confidence_drivers": top_drivers,
            "top_confidence_risks": top_risks,
            "allocation_confidence": cace_v1_report.get("allocation_confidence"),
            "strategy_confidence": cace_v1_report.get("strategy_confidence"),
            "portfolio_confidence": cace_v1_report.get("portfolio_confidence"),
        }
