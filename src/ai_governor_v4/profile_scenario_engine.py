"""Profile-switch scenario evaluation for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.health_forecaster import HealthForecaster
from src.ai_governor_v3.recovery_forecaster import RecoveryForecaster
from src.ai_governor_v3.risk_budget_forecaster import RiskBudgetForecaster
from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG


class ProfileScenarioEngine:
    def __init__(self, config: StrategicConfig | None = None) -> None:
        self._config = config or DEFAULT_STRATEGIC_CONFIG
        self._health = HealthForecaster()
        self._recovery = RecoveryForecaster()
        self._risk = RiskBudgetForecaster()

    def evaluate(self, context: GovernorContext, *, action_type: str) -> dict[str, dict[str, float]]:
        profile_boost = 0.0
        risk_reduction = 0.0
        if action_type == "PROFILE_SWITCH":
            if context.current_state == "recovery":
                profile_boost = 6.0
                risk_reduction = 0.15
            else:
                profile_boost = 4.0
                risk_reduction = 0.25

        health_fc = self._health.forecast(context)
        recovery_fc = self._recovery.forecast(context)
        risk_fc = self._risk.forecast(context)
        recovery_probs = recovery_fc.get("recovery_probability") or {}

        metrics: dict[str, dict[str, float]] = {}
        for days in self._config.forecast_days:
            hkey = f"{days}d"
            health = float((health_fc.get("future_health") or {}).get(hkey, context.health_score))
            health = min(100.0, health + profile_boost * (1.0 - days / 360.0))
            recovery_prob = float(recovery_probs.get(f"recovery_probability_{days}d") or 0.0)
            if context.current_state == "recovery":
                recovery_prob = max(0.0, recovery_prob - 0.15)
            risk_remaining = float((risk_fc.get("risk_budget_forecast") or {}).get(hkey, 50.0))
            dd = float((risk_fc.get("dd_forecast") or {}).get(hkey, context.dd_pct))
            dd = max(0.0, dd * (1.0 - risk_reduction))

            metrics[hkey] = self._metrics_row(
                context=context,
                health=health,
                recovery_prob=recovery_prob,
                risk_remaining=risk_remaining,
                dd=dd,
                days=days,
            )
        return metrics

    def _metrics_row(
        self,
        *,
        context: GovernorContext,
        health: float,
        recovery_prob: float,
        risk_remaining: float,
        dd: float,
        days: int,
    ) -> dict[str, float]:
        pf = min(3.5, max(0.5, context.portfolio_fit / 30.0 + health / 50.0))
        pass_prob = min(100.0, context.pass_rate + (health - context.health_score) * 0.2)
        expected_r = round(pf * (pass_prob / 100.0) * (1.0 - recovery_prob) * 2.0, 3)
        return {
            "expected_r": expected_r,
            "expected_pf": round(pf, 3),
            "expected_dd": round(dd, 2),
            "pass_probability": round(pass_prob, 2),
            "health_score": round(health, 1),
            "recovery_probability": round(recovery_prob, 3),
            "risk_budget_remaining": round(risk_remaining, 2),
            "horizon_days": float(days),
        }
