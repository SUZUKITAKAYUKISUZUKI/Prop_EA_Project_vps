"""Strategic score computation for AGE v4 branches."""
from __future__ import annotations

from typing import Any

from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG


class FutureScoreEngine:
    def __init__(self, config: StrategicConfig | None = None) -> None:
        self._config = config or DEFAULT_STRATEGIC_CONFIG

    def score_branch(
        self,
        metrics_by_horizon: dict[str, dict[str, float]],
        *,
        baseline_metrics: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        primary_key = f"{self._config.primary_horizon_days}d"
        metrics = metrics_by_horizon.get(primary_key) or next(iter(metrics_by_horizon.values()), {})
        baseline = (baseline_metrics or {}).get(primary_key) or {}

        health = float(metrics.get("health_score") or 0.0)
        pf = float(metrics.get("expected_pf") or 0.0)
        recovery = float(metrics.get("recovery_probability") or 0.0)
        dd = float(metrics.get("expected_dd") or 0.0)
        risk_budget = float(metrics.get("risk_budget_remaining") or 0.0)
        pass_prob = float(metrics.get("pass_probability") or 0.0)

        pf_norm = min(100.0, pf / 3.5 * 100.0)
        recovery_norm = (1.0 - recovery) * 100.0
        dd_protection = max(0.0, 100.0 - dd)
        weights = self._config.score_weights

        strategic_score = (
            weights["future_health"] * health
            + weights["expected_pf"] * pf_norm
            + weights["recovery_probability"] * recovery_norm
            + weights["dd_protection"] * dd_protection
            + weights["risk_budget"] * risk_budget
            + weights["pass_probability"] * pass_prob
        )
        strategic_score = round(min(100.0, max(0.0, strategic_score)), 1)

        baseline_score = self._score_metrics(baseline) if baseline else strategic_score
        expected_benefit = round(strategic_score - baseline_score, 2)
        expected_risk = round(
            recovery * 100.0 + max(0.0, dd - float(baseline.get("expected_dd") or dd)) * 2.0,
            2,
        )

        return {
            "strategic_score": strategic_score,
            "expected_benefit": expected_benefit,
            "expected_risk": expected_risk,
            "primary_horizon": primary_key,
            "components": {
                "future_health": round(health, 1),
                "expected_pf": round(pf_norm, 1),
                "recovery_probability": round(recovery_norm, 1),
                "dd_protection": round(dd_protection, 1),
                "risk_budget": round(risk_budget, 1),
                "pass_probability": round(pass_prob, 1),
            },
        }

    def _score_metrics(self, metrics: dict[str, float]) -> float:
        if not metrics:
            return 0.0
        result = self.score_branch({f"{self._config.primary_horizon_days}d": metrics})
        return float(result.get("strategic_score") or 0.0)
