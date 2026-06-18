"""Risk scoring from PRAE, State Analytics, AGE."""
from __future__ import annotations

from typing import Any


class RiskEngine:
    def evaluate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        prae = bundle.get("prae_v2") or {}
        state = bundle.get("state_analytics") or {}
        age = bundle.get("age_v4") or {}
        paae = bundle.get("paae") or {}

        risks = prae.get("strategy_risk") or []
        health = float((prae.get("health_report") or {}).get("health_score") or 70)
        max_dd = max((float(r.get("max_dd") or 0) for r in risks), default=0)
        avg_risk = sum(float(r.get("risk_score") or 0) for r in risks) / len(risks) if risks else 40
        recovery_prob = 100.0 - min(100.0, float(state.get("recovery_events") or 0) * 15.0)
        expected_dd = float((age.get("best_future_metrics") or {}).get("expected_dd") or 5)
        risk_budget_used = min(100.0, max_dd * 10.0 + expected_dd * 2.0)
        concentration = self._concentration(paae.get("current_weights") or {})
        fragility = min(100.0, avg_risk * 0.6 + concentration * 100.0 * 0.4)

        raw_risk = min(100.0, (avg_risk + risk_budget_used + fragility) / 3.0)
        risk_score = round(max(0.0, 100.0 - raw_risk * 0.7 + health * 0.3), 2)

        top_risk = self._top_risk(risks, concentration)
        risk_items = [
            {
                "strategy": r.get("strategy"),
                "risk_score": r.get("risk_score"),
                "max_dd": r.get("max_dd"),
            }
            for r in sorted(risks, key=lambda x: float(x.get("risk_score") or 0), reverse=True)[:5]
        ]

        return {
            "risk_score": risk_score,
            "dd_risk": round(max_dd, 2),
            "recovery_probability": round(recovery_prob, 2),
            "risk_budget_consumption": round(risk_budget_used, 2),
            "strategy_concentration": round(concentration, 3),
            "portfolio_fragility": round(fragility, 2),
            "top_risk": top_risk,
            "risks": risk_items,
        }

    def _concentration(self, weights: dict[str, Any]) -> float:
        if not weights:
            return 0.0
        values = [float(v) for v in weights.values()]
        if max(values) > 1.0:
            values = [v / 100.0 for v in values]
        return max(values)

    def _top_risk(self, risks: list[dict[str, Any]], concentration: float) -> str:
        if concentration >= 0.4:
            return "Strategy concentration rising"
        if risks:
            top = max(risks, key=lambda r: float(r.get("risk_score") or 0))
            return f"Elevated risk in {top.get('strategy')}"
        return "Risk within normal bounds"
