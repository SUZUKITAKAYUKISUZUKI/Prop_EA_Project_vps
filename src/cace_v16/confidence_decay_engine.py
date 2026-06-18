"""Confidence decay projection for CACE v1.6."""
from __future__ import annotations

import math
from typing import Any

from src.cace_v16.confidence_v16_models import (
    durability_category,
    half_life_from_stability,
)


class ConfidenceDecayEngine:
    """Estimate confidence durability over time — evaluation only."""

    def evaluate(self, cace_v15_report: dict[str, Any]) -> dict[str, Any]:
        current = float(cace_v15_report.get("confidence") or 0)
        stability = float(cace_v15_report.get("confidence_stability") or 70)
        half_life = half_life_from_stability(stability)

        forecast = {
            "today": round(current, 1),
            "30d": round(self._decay(current, 30, half_life), 1),
            "60d": round(self._decay(current, 60, half_life), 1),
            "90d": round(self._decay(current, 90, half_life), 1),
            "180d": round(self._decay(current, 180, half_life), 1),
        }
        durability_score = round(
            (forecast["30d"] + forecast["60d"] + forecast["90d"]) / 3.0,
            1,
        )

        decay_curve = [
            {"day": float(days), "confidence": round(self._decay(current, days, half_life), 1)}
            for days in (0, 30, 60, 90, 120, 180)
        ]

        return {
            "current_confidence": round(current, 1),
            "half_life": half_life,
            "durability_score": durability_score,
            "durability_category": durability_category(durability_score),
            "forecast": forecast,
            "decay_curve": decay_curve,
            "confidence_stability": round(stability, 1),
        }

    def _decay(self, current: float, days: int, half_life: int) -> float:
        if half_life <= 0:
            return max(0.0, current)
        return max(0.0, min(100.0, current * math.exp(-days / half_life)))
