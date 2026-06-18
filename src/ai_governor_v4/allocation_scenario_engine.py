"""Allocation-change scenario evaluation for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.allocation_forecaster import AllocationForecaster
from src.ai_governor_v4.profile_scenario_engine import ProfileScenarioEngine
from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG


class AllocationScenarioEngine:
    def __init__(self, config: StrategicConfig | None = None) -> None:
        self._config = config or DEFAULT_STRATEGIC_CONFIG
        self._allocation = AllocationForecaster()
        self._base = ProfileScenarioEngine(config)

    def evaluate(
        self,
        context: GovernorContext,
        *,
        action_type: str,
        modification: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        baseline = self._base.evaluate(context, action_type="DO_NOTHING")
        alloc_fc = self._allocation.forecast(context)
        max_drift = float(alloc_fc.get("max_current_drift_pct") or 0.0)
        drift_improvement = self._drift_improvement(action_type, modification, max_drift)

        metrics: dict[str, dict[str, float]] = {}
        for days in self._config.forecast_days:
            hkey = f"{days}d"
            base = dict(baseline.get(hkey) or {})
            decay = 1.0 - (days / 360.0) * 0.2
            benefit = drift_improvement * decay
            base["health_score"] = min(100.0, float(base.get("health_score") or 0) + benefit * 0.4)
            base["expected_dd"] = max(0.0, float(base.get("expected_dd") or 0) - benefit * 0.05)
            base["pass_probability"] = min(100.0, float(base.get("pass_probability") or 0) + benefit * 0.3)
            base["expected_pf"] = round(
                min(3.5, float(base.get("expected_pf") or 1.0) + benefit * 0.01),
                3,
            )
            base["risk_budget_remaining"] = min(
                100.0,
                float(base.get("risk_budget_remaining") or 0) + benefit * 0.2,
            )
            base["expected_r"] = round(
                float(base.get("expected_pf") or 1.0) * (float(base.get("pass_probability") or 0) / 100.0) * 2.0,
                3,
            )
            base["horizon_days"] = float(days)
            metrics[hkey] = base
        return metrics

    def _drift_improvement(
        self,
        action_type: str,
        modification: dict[str, Any],
        max_drift: float,
    ) -> float:
        if action_type not in {"REDUCE_ALLOCATION", "INCREASE_ALLOCATION"}:
            return 0.0
        delta = modification.get("allocation_delta") or {}
        magnitude = sum(abs(float(v)) for v in delta.values()) * 100.0
        return min(max_drift, magnitude + max_drift * 0.3)
