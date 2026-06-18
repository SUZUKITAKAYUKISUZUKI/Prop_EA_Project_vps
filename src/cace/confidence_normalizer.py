"""Normalize and categorize CACE confidence scores."""
from __future__ import annotations

from src.cace.confidence_factors import RANK_THRESHOLDS


class ConfidenceNormalizer:
    @staticmethod
    def clamp(value: float, *, low: float = 0.0, high: float = 100.0) -> float:
        return round(min(high, max(low, value)), 1)

    @staticmethod
    def category(score: float) -> str:
        for threshold, label in RANK_THRESHOLDS:
            if score >= threshold:
                return label
        return "VERY_LOW"

    @staticmethod
    def normalize_variance(values: list[float], *, invert: bool = False) -> float:
        if not values:
            return 50.0
        if len(values) == 1:
            return 80.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        spread = variance ** 0.5
        stability = max(0.0, 100.0 - spread * 2.0)
        if invert:
            return ConfidenceNormalizer.clamp(100.0 - stability)
        return ConfidenceNormalizer.clamp(stability)

    @staticmethod
    def normalize_agreement(values: list[float]) -> float:
        if not values:
            return 50.0
        if len(values) == 1:
            return 75.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 50.0
        spread = max(values) - min(values)
        rel_spread = spread / max(abs(mean), 1.0)
        return ConfidenceNormalizer.clamp(100.0 - rel_spread * 100.0)
