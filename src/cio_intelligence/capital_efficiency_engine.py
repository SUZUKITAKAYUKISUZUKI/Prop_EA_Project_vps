"""Capital efficiency scoring."""
from __future__ import annotations

from typing import Any


class CapitalEfficiencyEngine:
    def evaluate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        paae = bundle.get("paae") or {}
        slm = bundle.get("slm") or {}
        prae = bundle.get("prae_v2") or {}

        current = paae.get("current_weights") or paae.get("strategy_allocations") or {}
        recommended = paae.get("recommended_weights") or {}
        usage = self._usage(current)
        unused_capacity = round(max(0.0, 1.0 - usage) * 100.0, 2)
        utilization = self._strategy_utilization(slm)
        allocation_efficiency = self._allocation_efficiency(current, recommended)
        health = float((prae.get("health_report") or {}).get("health_score") or 70)

        capital_efficiency = round(
            usage * 100.0 * 0.25
            + utilization * 0.30
            + allocation_efficiency * 0.25
            + health * 0.20,
            2,
        )

        return {
            "capital_efficiency": capital_efficiency,
            "current_capital_usage_pct": round(usage * 100.0, 2),
            "unused_capacity_pct": unused_capacity,
            "strategy_utilization": round(utilization, 2),
            "allocation_efficiency": round(allocation_efficiency, 2),
        }

    def _usage(self, weights: dict[str, Any]) -> float:
        total = sum(float(v) for v in weights.values())
        if total <= 0:
            return 0.0
        if total > 1.0:
            return min(1.0, total / 100.0)
        return min(1.0, total)

    def _strategy_utilization(self, slm: dict[str, Any]) -> float:
        strategies = slm.get("strategies") or []
        if not strategies:
            return 70.0
        active = sum(1 for s in strategies if str(s.get("stage", "")).upper() not in {"RETIRED", "INACTIVE"})
        fits = [float(s.get("portfolio_fit_score") or 70) for s in strategies]
        return min(100.0, (active / len(strategies)) * 50.0 + sum(fits) / len(fits) * 0.5)

    def _allocation_efficiency(self, current: dict[str, Any], recommended: dict[str, Any]) -> float:
        if not current or not recommended:
            return 75.0
        drift = sum(abs(float(recommended.get(k, 0)) - float(current.get(k, 0))) for k in set(current) | set(recommended))
        return max(0.0, min(100.0, 100.0 - drift * 200.0))
