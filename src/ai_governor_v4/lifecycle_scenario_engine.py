"""Lifecycle-change scenario evaluation for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.lifecycle_forecaster import LifecycleForecaster
from src.ai_governor_v4.profile_scenario_engine import ProfileScenarioEngine
from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG


class LifecycleScenarioEngine:
    def __init__(self, config: StrategicConfig | None = None) -> None:
        self._config = config or DEFAULT_STRATEGIC_CONFIG
        self._lifecycle = LifecycleForecaster()
        self._base = ProfileScenarioEngine(config)

    def evaluate(
        self,
        context: GovernorContext,
        *,
        action_type: str,
        modification: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        baseline = self._base.evaluate(context, action_type="DO_NOTHING")
        lifecycle_fc = self._lifecycle.forecast(context)
        code = (
            modification.get("promote_strategy")
            or modification.get("demote_strategy")
            or ""
        )
        strategy_row = (lifecycle_fc.get("strategies") or {}).get(code) or {}
        current_block = strategy_row.get("current") or {}
        fit_delta = self._fit_delta(action_type, current_block, strategy_row)

        metrics: dict[str, dict[str, float]] = {}
        for days in self._config.forecast_days:
            hkey = f"{days}d"
            base = dict(baseline.get(hkey) or {})
            projected_fit = float((strategy_row.get(f"{days}d") or {}).get("portfolio_fit") or 0)
            if projected_fit > 0:
                fit_delta = max(fit_delta, (projected_fit - float(strategy_row.get("current_fit") or 0)) * 0.5)

            if action_type == "PROMOTE_STRATEGY":
                base["health_score"] = min(100.0, float(base.get("health_score") or 0) + fit_delta * 0.5)
                base["expected_pf"] = round(min(3.5, float(base.get("expected_pf") or 1.0) + fit_delta * 0.02), 3)
                base["recovery_probability"] = max(
                    0.0,
                    float(base.get("recovery_probability") or 0) - 0.05,
                )
            elif action_type == "DEMOTE_STRATEGY":
                base["health_score"] = max(0.0, float(base.get("health_score") or 0) - abs(fit_delta) * 0.3)
                base["expected_dd"] = min(100.0, float(base.get("expected_dd") or 0) + 1.0)
                base["recovery_probability"] = min(
                    1.0,
                    float(base.get("recovery_probability") or 0) + 0.08,
                )

            base["pass_probability"] = min(
                100.0,
                float(base.get("pass_probability") or 0) + fit_delta * 0.15,
            )
            base["expected_r"] = round(
                float(base.get("expected_pf") or 1.0) * (float(base.get("pass_probability") or 0) / 100.0) * 2.0,
                3,
            )
            base["horizon_days"] = float(days)
            metrics[hkey] = base
        return metrics

    def _fit_delta(
        self,
        action_type: str,
        current_block: dict[str, Any],
        strategy_row: dict[str, Any],
    ) -> float:
        current_fit = float(current_block.get("portfolio_fit") or 50.0)
        if action_type == "PROMOTE_STRATEGY":
            return max(5.0, 85.0 - current_fit) * 0.3
        if action_type == "DEMOTE_STRATEGY":
            return -max(5.0, current_fit - 40.0) * 0.2
        return 0.0
