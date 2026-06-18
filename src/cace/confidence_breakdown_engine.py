"""Confidence component breakdown for CACE v1.5."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_v15_config import BREAKDOWN_KEYS


class ConfidenceBreakdownEngine:
    def build(self, factors: dict[str, float] | Any) -> dict[str, float]:
        if hasattr(factors, "to_dict"):
            source = factors.to_dict()
        else:
            source = dict(factors or {})

        breakdown = {
            "historical_reliability": round(float(source.get("historical_reliability") or 0), 1),
            "monte_carlo_stability": round(float(source.get("monte_carlo_stability") or 0), 1),
            "forecast_stability": round(float(source.get("forecast_stability") or 0), 1),
            "portfolio_health": round(float(source.get("portfolio_health") or 0), 1),
            "lifecycle_quality": round(float(source.get("lifecycle_quality") or 0), 1),
        }
        return breakdown

    def top_drivers(self, breakdown: dict[str, float], *, limit: int = 3) -> list[str]:
        labels = {
            "historical_reliability": "Historical allocation reliability",
            "monte_carlo_stability": "Monte Carlo stability",
            "forecast_stability": "Forecast horizon agreement",
            "portfolio_health": "Portfolio health",
            "lifecycle_quality": "Strategy lifecycle quality",
        }
        ranked = sorted(
            ((k, breakdown.get(k, 0)) for k in BREAKDOWN_KEYS),
            key=lambda item: item[1],
            reverse=True,
        )
        return [labels.get(key, key) for key, score in ranked[:limit] if score >= 60]

    def top_risks(self, breakdown: dict[str, float], *, limit: int = 3) -> list[str]:
        labels = {
            "historical_reliability": "Weak historical allocation reliability",
            "monte_carlo_stability": "High Monte Carlo variance",
            "forecast_stability": "Forecast horizon disagreement",
            "portfolio_health": "Portfolio health below trust threshold",
            "lifecycle_quality": "Lifecycle quality drag on confidence",
        }
        ranked = sorted(
            ((k, breakdown.get(k, 0)) for k in BREAKDOWN_KEYS),
            key=lambda item: item[1],
        )
        return [labels.get(key, key) for key, score in ranked[:limit] if score < 55]
