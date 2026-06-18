"""Confidence factor weights and categories for CACE."""
from __future__ import annotations

from dataclasses import dataclass, field


COMPONENT_WEIGHTS: dict[str, float] = {
    "historical_reliability": 0.30,
    "monte_carlo_stability": 0.25,
    "forecast_stability": 0.20,
    "portfolio_health": 0.15,
    "lifecycle_quality": 0.10,
}

RANK_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (95.0, "EXTREME"),
    (85.0, "VERY_HIGH"),
    (70.0, "HIGH"),
    (55.0, "MODERATE"),
    (40.0, "LOW"),
    (0.0, "VERY_LOW"),
)

CACHE_TTL_SECONDS = 3600
CACHE_KEY_ALLOCATION = "cace:allocation:{profile_id}"
CACHE_KEY_STRATEGY = "cace:strategy:{strategy}"
CACHE_KEY_PORTFOLIO = "cace:portfolio:{profile_id}"


@dataclass
class ConfidenceFactors:
    historical_reliability: float = 50.0
    monte_carlo_stability: float = 50.0
    forecast_stability: float = 50.0
    portfolio_health: float = 50.0
    lifecycle_quality: float = 50.0
    weights: dict[str, float] = field(default_factory=lambda: dict(COMPONENT_WEIGHTS))

    def to_dict(self) -> dict[str, float]:
        return {
            "historical_reliability": round(self.historical_reliability, 1),
            "monte_carlo_stability": round(self.monte_carlo_stability, 1),
            "forecast_stability": round(self.forecast_stability, 1),
            "portfolio_health": round(self.portfolio_health, 1),
            "lifecycle_quality": round(self.lifecycle_quality, 1),
        }
