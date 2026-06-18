"""Risk threat detection for APM v1."""
from __future__ import annotations

from typing import Any

from src.apm.executive_context import ExecutiveContext


class RiskEngine:
    def evaluate(self, context: ExecutiveContext) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        health = float(context.health.get("health_score") or 100)

        for row in context.risk.get("strategy_risk") or []:
            code = str(row.get("strategy") or "")
            risk_score = float(row.get("risk_score") or 0)
            if risk_score < 70:
                continue
            dd = float(row.get("max_dd") or row.get("dd_contribution") or 0)
            alerts.append(
                {
                    "strategy": code,
                    "risk_score": risk_score,
                    "dd_contribution_pct": round(dd * 10 if dd < 15 else dd, 1),
                    "health_impact": round(-(risk_score / 100.0) * (100.0 - health) * 0.5, 1),
                    "alert_type": "RISK_ALERT",
                    "message": f"{code}: risk {risk_score:.0f}, DD contribution elevated.",
                }
            )

        for row in context.strategy_lifecycle.get("retirement_candidates") or []:
            code = str(row.get("strategy") or "")
            alerts.append(
                {
                    "strategy": code,
                    "risk_score": float(row.get("score") or 55),
                    "dd_contribution_pct": 0.0,
                    "health_impact": -8.0,
                    "alert_type": "RISK_ALERT",
                    "message": f"{code}: retirement candidate — lifecycle risk elevated.",
                }
            )
        return alerts
