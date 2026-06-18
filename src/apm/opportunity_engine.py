"""Opportunity detection for APM v1."""
from __future__ import annotations

from typing import Any

from src.apm.executive_context import ExecutiveContext


class OpportunityEngine:
    def evaluate(self, context: ExecutiveContext) -> list[dict[str, Any]]:
        opportunities: list[dict[str, Any]] = []
        current = context.allocation.get("current_weights") or {}
        recommended = context.allocation.get("recommended_weights") or {}

        for strategy in context.strategy_lifecycle.get("promotion_candidates") or []:
            code = str(strategy.get("strategy") or "")
            if not code:
                continue
            opportunities.append(
                {
                    "strategy": code,
                    "portfolio_fit": float(strategy.get("portfolio_fit_score") or strategy.get("score") or 0),
                    "lifecycle_score": float(strategy.get("score") or 0),
                    "current_allocation_pct": self._pct(current.get(code, 0)),
                    "recommended_allocation_pct": self._pct(recommended.get(code, current.get(code, 0))),
                    "alert_type": "OPPORTUNITY_ALERT",
                    "message": f"Strategy {code} promotion opportunity detected.",
                }
            )

        for row in context.strategy_lifecycle.get("strategies") or []:
            code = str(row.get("strategy") or "")
            fit = float(row.get("portfolio_fit_score") or 0)
            cur = self._pct(current.get(code, 0))
            rec = self._pct(recommended.get(code, cur))
            if fit >= 85 and rec > cur + 5:
                opportunities.append(
                    {
                        "strategy": code,
                        "portfolio_fit": fit,
                        "lifecycle_score": float(row.get("score") or fit),
                        "current_allocation_pct": cur,
                        "recommended_allocation_pct": rec,
                        "alert_type": "OPPORTUNITY_ALERT",
                        "message": f"Strategy {code}: fit {fit:.0f}, allocation {cur:.0f}% → {rec:.0f}%.",
                    }
                )
        return opportunities

    def _pct(self, value: Any) -> float:
        v = float(value or 0)
        return round(v * 100.0 if v <= 1.0 else v, 1)
