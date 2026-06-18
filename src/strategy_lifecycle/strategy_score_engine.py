"""Composite strategy score for SLM (0-100)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyScoreInput:
    oos_pf: float = 0.0
    pass_rate: float = 0.0
    total_r: float = 0.0
    stability: float = 0.0
    health_impact: float = 50.0
    portfolio_fit: float = 50.0


class StrategyScoreEngine:
    WEIGHTS = {
        "oos_pf": 0.25,
        "pass_rate": 0.20,
        "total_r": 0.15,
        "stability": 0.15,
        "health_impact": 0.10,
        "portfolio_fit": 0.15,
    }

    def _norm(self, value: float, cap: float) -> float:
        if cap <= 0:
            return 0.0
        return max(0.0, min(100.0, float(value) / cap * 100.0))

    def compute(self, metrics: StrategyScoreInput | dict[str, Any]) -> float:
        if isinstance(metrics, dict):
            metrics = StrategyScoreInput(
                oos_pf=float(metrics.get("oos_pf") or 0.0),
                pass_rate=float(metrics.get("pass_rate") or 0.0),
                total_r=float(metrics.get("total_r") or 0.0),
                stability=float(metrics.get("stability") or 0.0),
                health_impact=float(metrics.get("health_impact") or 50.0),
                portfolio_fit=float(
                    metrics.get("portfolio_fit")
                    or metrics.get("portfolio_fit_score")
                    or 50.0
                ),
            )
        parts = {
            "oos_pf": self._norm(min(metrics.oos_pf, 3.0), 3.0),
            "pass_rate": self._norm(metrics.pass_rate, 100.0),
            "total_r": self._norm(max(metrics.total_r, 0.0), 200.0),
            "stability": self._norm(metrics.stability, 100.0),
            "health_impact": self._norm(metrics.health_impact, 100.0),
            "portfolio_fit": self._norm(metrics.portfolio_fit, 100.0),
        }
        score = sum(self.WEIGHTS[k] * parts[k] for k in self.WEIGHTS)
        return round(score, 1)
