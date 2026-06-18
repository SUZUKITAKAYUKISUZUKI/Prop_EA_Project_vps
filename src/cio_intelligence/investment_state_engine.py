"""Investment state classification for CIL v1."""
from __future__ import annotations

from typing import Any


class InvestmentStateEngine:
    def evaluate(self, bundle: dict[str, Any]) -> list[str]:
        states: list[str] = []
        paae = bundle.get("paae") or {}
        prae = bundle.get("prae_v2") or {}
        state = bundle.get("state_analytics") or {}
        slm = bundle.get("slm") or {}
        age = bundle.get("age_v4") or {}

        current = paae.get("current_weights") or paae.get("strategy_allocations") or {}
        recommended = paae.get("recommended_weights") or {}
        drift = self._allocation_drift(current, recommended)
        if drift > 0.12:
            states.append("UNDER_ALLOCATED" if self._total(recommended) > self._total(current) else "OVER_ALLOCATED")
        else:
            states.append("BALANCED")

        health = float((prae.get("health_report") or {}).get("health_score") or 70)
        pass_prob = float((age.get("best_future_metrics") or {}).get("pass_probability") or 85)
        if pass_prob >= 90 and health >= 75:
            states.append("HIGH_GROWTH")

        risks = prae.get("strategy_risk") or []
        max_risk = max((float(r.get("risk_score") or 0) for r in risks), default=0)
        if max_risk >= 75 or str(state.get("current_state") or "").lower() == "recovery":
            states.append("HIGH_RISK")
        if str(state.get("current_state") or "").lower() == "recovery":
            states.append("RECOVERY_MODE")

        efficiency_hint = self._capital_usage(current)
        if efficiency_hint >= 0.85 and health >= 70:
            states.append("CAPITAL_EFFICIENT")

        concentration = self._concentration(current)
        if concentration >= 0.45:
            states.append("STRATEGY_CONCENTRATED")
        elif len(current) >= 3 and concentration <= 0.35:
            states.append("DIVERSIFIED")

        return list(dict.fromkeys(states))

    def _allocation_drift(self, current: dict[str, Any], recommended: dict[str, Any]) -> float:
        keys = set(current) | set(recommended)
        return sum(abs(float(recommended.get(k, 0)) - float(current.get(k, 0))) for k in keys)

    def _total(self, weights: dict[str, Any]) -> float:
        return sum(float(v) for v in weights.values())

    def _capital_usage(self, weights: dict[str, Any]) -> float:
        total = self._total(weights)
        return min(1.0, total) if total <= 1.0 else min(1.0, total / 100.0)

    def _concentration(self, weights: dict[str, Any]) -> float:
        if not weights:
            return 0.0
        values = [float(v) for v in weights.values()]
        if max(values) > 1.0:
            values = [v / 100.0 for v in values]
        return max(values)
