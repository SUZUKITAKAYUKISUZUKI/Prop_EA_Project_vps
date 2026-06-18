"""Allocation drift forecast for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG


class AllocationForecaster:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG

    def forecast(self, context: GovernorContext) -> dict[str, Any]:
        current = context.current_allocation
        recommended = context.recommended_allocation or context.paae.get("recommended_weights") or {}
        all_codes = set(current) | set(recommended)
        strategies: dict[str, Any] = {}
        warnings: list[str] = []
        max_drift = 0.0

        for code in sorted(all_codes):
            cur_w = float(current.get(code, 0.0))
            rec_w = float(recommended.get(code, 0.0))
            drift = abs(cur_w - rec_w)
            drift_rate = _drift_rate(context, code, cur_w, rec_w)
            projections: dict[str, Any] = {
                "current_pct": round(cur_w * 100.0, 1),
                "recommended_pct": round(rec_w * 100.0, 1),
                "drift_pct": round(drift * 100.0, 1),
            }
            for days in self._config.forecast_days:
                projected_rec = max(0.0, min(1.0, rec_w + drift_rate * (days / 30.0)))
                projected_drift = abs(cur_w - projected_rec)
                projections[f"{days}d"] = {
                    "recommended_pct": round(projected_rec * 100.0, 1),
                    "drift_pct": round(projected_drift * 100.0, 1),
                }
                if projected_drift * 100.0 >= 10.0:
                    warnings.append(f"ALLOCATION_DRIFT_WARNING:{code}:{days}d")

            max_drift = max(max_drift, drift * 100.0)
            strategies[code] = projections

        return {
            "strategies": strategies,
            "max_current_drift_pct": round(max_drift, 1),
            "warnings": warnings,
            "overweight_risk": _overweight_codes(context),
            "underweight_opportunities": _underweight_codes(context),
        }


def _drift_rate(context: GovernorContext, code: str, current: float, recommended: float) -> float:
    gap = recommended - current
    paae_drift = context.paae.get("drift_alerts") or []
    for alert in paae_drift:
        if str(alert.get("strategy")) == code:
            return gap * 0.5 + float(alert.get("drift") or 0) * 0.1
    return gap * 0.25


def _overweight_codes(context: GovernorContext) -> list[str]:
    current = context.current_allocation
    recommended = context.recommended_allocation
    return [
        code
        for code in current
        if float(current.get(code, 0)) > float(recommended.get(code, 0)) + 0.05
    ]


def _underweight_codes(context: GovernorContext) -> list[str]:
    current = context.current_allocation
    recommended = context.recommended_allocation
    return [
        code
        for code in recommended
        if float(recommended.get(code, 0)) > float(current.get(code, 0)) + 0.05
    ]
