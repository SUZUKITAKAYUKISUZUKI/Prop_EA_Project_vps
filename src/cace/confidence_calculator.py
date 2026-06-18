"""Weighted confidence score calculator for CACE."""
from __future__ import annotations

from src.cace.confidence_factors import ConfidenceFactors
from src.cace.confidence_normalizer import ConfidenceNormalizer


class ConfidenceCalculator:
    def __init__(self, normalizer: ConfidenceNormalizer | None = None) -> None:
        self._normalizer = normalizer or ConfidenceNormalizer()

    def compute(self, factors: ConfidenceFactors) -> float:
        weights = factors.weights
        score = (
            weights["historical_reliability"] * factors.historical_reliability
            + weights["monte_carlo_stability"] * factors.monte_carlo_stability
            + weights["forecast_stability"] * factors.forecast_stability
            + weights["portfolio_health"] * factors.portfolio_health
            + weights["lifecycle_quality"] * factors.lifecycle_quality
        )
        return self._normalizer.clamp(score)

    def category(self, score: float) -> str:
        return self._normalizer.category(score)
