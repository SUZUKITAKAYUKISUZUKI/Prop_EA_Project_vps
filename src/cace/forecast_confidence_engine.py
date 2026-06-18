"""Forecast horizon agreement confidence for CACE."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_normalizer import ConfidenceNormalizer


class ForecastConfidenceEngine:
    def __init__(self, normalizer: ConfidenceNormalizer | None = None) -> None:
        self._normalizer = normalizer or ConfidenceNormalizer()

    def score(self, *, age_v4: dict[str, Any], age_v3: dict[str, Any] | None = None) -> float:
        horizons = ("30d", "60d", "90d", "180d")
        health_values: list[float] = []
        dd_values: list[float] = []
        pass_values: list[float] = []

        best_metrics = age_v4.get("best_future_metrics") or {}
        if best_metrics:
            health_values.append(float(best_metrics.get("health_score") or 0))
            dd_values.append(float(best_metrics.get("expected_dd") or 0))
            pass_values.append(float(best_metrics.get("pass_probability") or 0))

        for scenario in age_v4.get("future_scenarios") or []:
            if scenario.get("action_type") != "DO_NOTHING":
                continue
            by_h = scenario.get("metrics_by_horizon") or {}
            for h in horizons:
                metrics = by_h.get(h) or {}
                if metrics.get("health_score") is not None:
                    health_values.append(float(metrics["health_score"]))
                if metrics.get("expected_dd") is not None:
                    dd_values.append(float(metrics["expected_dd"]))
                if metrics.get("pass_probability") is not None:
                    pass_values.append(float(metrics["pass_probability"]))
            break

        if age_v3:
            future_health = (age_v3.get("health_forecast") or {}).get("future_health") or {}
            for h in horizons:
                if future_health.get(h) is not None:
                    health_values.append(float(future_health[h]))

        agreements = []
        if len(health_values) >= 2:
            agreements.append(self._normalizer.normalize_agreement(health_values))
        if len(dd_values) >= 2:
            agreements.append(self._normalizer.normalize_agreement(dd_values))
        if len(pass_values) >= 2:
            agreements.append(self._normalizer.normalize_agreement(pass_values))

        if not agreements:
            strategic = float(age_v4.get("strategic_score") or age_v4.get("future_score") or 50)
            return self._normalizer.clamp(strategic * 0.9)

        return self._normalizer.clamp(sum(agreements) / len(agreements))
