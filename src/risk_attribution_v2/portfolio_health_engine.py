"""Portfolio health report synthesizing PRAE v2 analytics."""
from __future__ import annotations

from typing import Any


class PortfolioHealthEngine:
    def build_report(
        self,
        *,
        strategy_risk: list[dict[str, Any]],
        symbol_risk: list[dict[str, Any]],
        recovery_analysis: list[dict[str, Any]],
        dd_attribution: dict[str, Any],
        state_health_score: float | None = None,
    ) -> dict[str, Any]:
        highest_risk_strategy = strategy_risk[0]["strategy"] if strategy_risk else None
        highest_risk_symbol = symbol_risk[0]["symbol"] if symbol_risk else None

        dd_contrib = dd_attribution.get("strategy_contribution") or {}
        worst_dd_contributor = max(dd_contrib, key=dd_contrib.get) if dd_contrib else highest_risk_strategy

        best_risk_adjusted = None
        best_score = -1.0
        for row in strategy_risk:
            adj = float(row.get("total_r", 0.0)) / max(float(row.get("risk_score", 1.0)), 1.0)
            if adj > best_score:
                best_score = adj
                best_risk_adjusted = row.get("strategy")

        base = float(state_health_score) if state_health_score is not None else 85.0
        recovery_penalty = min(20.0, len(recovery_analysis) * 3.0)
        risk_penalty = min(15.0, (strategy_risk[0]["risk_score"] / 100.0 * 15.0) if strategy_risk else 0.0)
        health_score = round(max(0.0, min(100.0, base - recovery_penalty - risk_penalty * 0.5)), 1)

        if health_score >= 90.0:
            status = "EXCELLENT"
        elif health_score >= 75.0:
            status = "GOOD"
        elif health_score >= 60.0:
            status = "FAIR"
        else:
            status = "AT_RISK"

        return {
            "health_score": health_score,
            "health_status": status,
            "highest_risk_strategy": highest_risk_strategy,
            "highest_risk_symbol": highest_risk_symbol,
            "recovery_events": len(recovery_analysis),
            "worst_dd_contributor": worst_dd_contributor,
            "best_risk_adjusted_strategy": best_risk_adjusted,
        }
