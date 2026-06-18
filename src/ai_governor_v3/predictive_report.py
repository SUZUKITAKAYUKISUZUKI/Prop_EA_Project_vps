"""Predictive governor report builder."""
from __future__ import annotations

from typing import Any


class PredictiveReport:
    def build(
        self,
        *,
        forecasts: dict[str, Any],
        timeline: list[dict[str, Any]],
        scenario_projection: dict[str, Any],
        alerts: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        age_v2_report: dict[str, Any],
        profile_id: str,
    ) -> dict[str, Any]:
        avg_confidence = (
            round(sum(r.get("confidence", 0) for r in recommendations) / len(recommendations), 1)
            if recommendations
            else 0.0
        )
        return {
            "profile_id": profile_id,
            "current_state": age_v2_report.get("current_state"),
            "current_profile": age_v2_report.get("current_profile"),
            "health_forecast": forecasts.get("health"),
            "recovery_forecast": forecasts.get("recovery"),
            "risk_budget_forecast": forecasts.get("risk_budget"),
            "strategy_forecast": forecasts.get("lifecycle"),
            "allocation_forecast": forecasts.get("allocation"),
            "profile_transition_forecast": forecasts.get("profile"),
            "future_state_timeline": timeline,
            "scenario_projection": scenario_projection,
            "predictive_alerts": alerts,
            "predictive_recommendations": recommendations,
            "recommended_actions": [r for r in recommendations if r.get("action") != "NO_ACTION"],
            "confidence": avg_confidence,
            "forecast_horizons": forecasts.get("forecast_horizons"),
            "age_v2_summary": {
                "health_score": age_v2_report.get("health_score"),
                "health_status": age_v2_report.get("health_status"),
                "risk_level": age_v2_report.get("risk_level"),
                "open_alerts": age_v2_report.get("open_alerts"),
            },
        }
