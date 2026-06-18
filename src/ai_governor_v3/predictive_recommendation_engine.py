"""Predictive governance recommendations for AGE v3."""
from __future__ import annotations

from enum import Enum
from typing import Any

from src.ai_governor.governor_context import GovernorContext


class PredictiveAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    REDUCE_RISK = "REDUCE_RISK"
    INCREASE_DIVERSIFICATION = "INCREASE_DIVERSIFICATION"
    PREPARE_RECOVERY = "PREPARE_RECOVERY"
    REBALANCE_EARLY = "REBALANCE_EARLY"
    PROMOTE_STRATEGY = "PROMOTE_STRATEGY"
    DEMOTE_STRATEGY = "DEMOTE_STRATEGY"
    RETIRE_STRATEGY = "RETIRE_STRATEGY"
    PROFILE_SWITCH = "PROFILE_SWITCH"


class PredictiveRecommendationEngine:
    def build(
        self,
        context: GovernorContext,
        forecasts: dict[str, Any],
        alerts: list[dict[str, Any]],
        *,
        age_v2_report: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []
        age_v2 = age_v2_report or {}

        for alert in alerts:
            rec = self._from_alert(alert, context, forecasts)
            if rec:
                rec["confidence"] = self._confidence(rec, context, age_v2, forecasts)
                recommendations.append(rec)

        recommendations.extend(self._lifecycle_recommendations(forecasts.get("lifecycle") or {}, context, age_v2))
        recommendations.extend(self._allocation_recommendations(forecasts.get("allocation") or {}, context, age_v2))
        recommendations.extend(self._recovery_recommendations(forecasts.get("recovery") or {}, context, age_v2))

        if not recommendations:
            recommendations.append(
                {
                    "action": PredictiveAction.NO_ACTION.value,
                    "priority": "INFO",
                    "confidence": 88.0,
                    "reason": "No significant deterioration projected within forecast horizons",
                    "reason_json": {"trigger": "stable_forecast"},
                    "expected_benefit": 0.0,
                    "expected_risk": 0.0,
                }
            )

        recommendations.sort(key=lambda r: (_priority_rank(r.get("priority", "LOW")), -float(r.get("confidence", 0))))
        return recommendations

    def _from_alert(self, alert: dict[str, Any], context: GovernorContext, forecasts: dict[str, Any]) -> dict[str, Any] | None:
        atype = str(alert.get("alert_type") or "")
        details = alert.get("details_json") or {}
        mapping = {
            "HEALTH_DETERIORATION": (PredictiveAction.REDUCE_RISK, "HIGH"),
            "RISK_DETERIORATION": (PredictiveAction.REDUCE_RISK, "CRITICAL"),
            "RISK_BUDGET_EXHAUSTION": (PredictiveAction.PREPARE_RECOVERY, "CRITICAL"),
            "FIT_DETERIORATION": (PredictiveAction.DEMOTE_STRATEGY, "MEDIUM"),
            "ALLOCATION_DRIFT_WARNING": (PredictiveAction.REBALANCE_EARLY, "MEDIUM"),
            "ALLOCATION_DETERIORATION": (PredictiveAction.REBALANCE_EARLY, "MEDIUM"),
            "PROFILE_TRANSITION_RISK": (PredictiveAction.PROFILE_SWITCH, "HIGH"),
        }
        for prefix, (action, priority) in mapping.items():
            if atype.startswith(prefix) or atype == prefix:
                return {
                    "action": action.value,
                    "priority": priority,
                    "reason": f"Predictive alert: {atype}",
                    "reason_json": {"trigger": atype, **details},
                    "strategy_id": details.get("strategy"),
                    "expected_benefit": 10.0,
                    "expected_risk": 8.0 if priority != "CRITICAL" else 20.0,
                }
        return None

    def _lifecycle_recommendations(
        self,
        lifecycle: dict[str, Any],
        context: GovernorContext,
        age_v2: dict[str, Any],
    ) -> list[dict[str, Any]]:
        recs = []
        for sid in lifecycle.get("promotion_candidates") or []:
            recs.append(_rec(PredictiveAction.PROMOTE_STRATEGY, sid, "Projected fit remains strong", "MEDIUM", 8, 4))
        for sid in lifecycle.get("demotion_candidates") or []:
            recs.append(_rec(PredictiveAction.DEMOTE_STRATEGY, sid, "Projected fit decline", "HIGH", 12, 10))
        for sid in lifecycle.get("retirement_candidates") or []:
            recs.append(_rec(PredictiveAction.RETIRE_STRATEGY, sid, "Projected retirement threshold", "HIGH", 15, 12))
        if context.portfolio_fit < 65 and lifecycle.get("demotion_candidates"):
            recs.append(
                {
                    "action": PredictiveAction.INCREASE_DIVERSIFICATION.value,
                    "priority": "MEDIUM",
                    "confidence": 72.0,
                    "reason": "Portfolio fit may decline — diversify allocation",
                    "reason_json": {"trigger": "fit_diversification"},
                    "expected_benefit": 8.0,
                    "expected_risk": 5.0,
                }
            )
        for r in recs:
            r["confidence"] = self._confidence(r, context, age_v2, {"lifecycle": lifecycle})
        return recs

    def _allocation_recommendations(
        self,
        allocation: dict[str, Any],
        context: GovernorContext,
        age_v2: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if float(allocation.get("max_current_drift_pct") or 0) < 8.0:
            return []
        rec = {
            "action": PredictiveAction.REBALANCE_EARLY.value,
            "priority": "MEDIUM",
            "reason": f"Projected allocation drift {allocation.get('max_current_drift_pct')}%",
            "reason_json": {
                "trigger": "allocation_drift_forecast",
                "underweight": allocation.get("underweight_opportunities"),
                "overweight": allocation.get("overweight_risk"),
            },
            "expected_benefit": float(allocation.get("max_current_drift_pct") or 0),
            "expected_risk": 4.0,
        }
        rec["confidence"] = self._confidence(rec, context, age_v2, {"allocation": allocation})
        return [rec]

    def _recovery_recommendations(
        self,
        recovery: dict[str, Any],
        context: GovernorContext,
        age_v2: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prob_90 = float((recovery.get("recovery_probability") or {}).get("recovery_probability_90d") or 0)
        if prob_90 < 0.25 or context.current_state == "recovery":
            return []
        rec = {
            "action": PredictiveAction.PREPARE_RECOVERY.value,
            "priority": "HIGH" if prob_90 >= 0.35 else "MEDIUM",
            "reason": f"Recovery probability 90d = {prob_90:.0%}",
            "reason_json": {"trigger": "recovery_probability", **recovery.get("recovery_probability", {})},
            "expected_benefit": 15.0,
            "expected_risk": prob_90 * 30.0,
        }
        rec["confidence"] = self._confidence(rec, context, age_v2, {"recovery": recovery})
        return [rec]

    def _confidence(
        self,
        rec: dict[str, Any],
        context: GovernorContext,
        age_v2: dict[str, Any],
        forecasts: dict[str, Any],
    ) -> float:
        votes = 0
        total = 4
        age_v2_actions = {a.get("action") or a.get("decision_type") for a in age_v2.get("recommended_actions") or []}
        if rec.get("action") in age_v2_actions or _maps_to_age_v2(rec.get("action")) in age_v2_actions:
            votes += 1
        if context.pdts.get("scenario_comparison"):
            votes += 1
        if context.paae.get("recommended_weights"):
            votes += 1
        if context.slm.get("strategies"):
            votes += 1
        base = 55.0 + votes / total * 35.0
        if rec.get("priority") == "CRITICAL":
            base += 5.0
        return round(min(100.0, base), 1)


def _rec(action: PredictiveAction, sid: str, reason: str, priority: str, benefit: float, risk: float) -> dict[str, Any]:
    return {
        "action": action.value,
        "priority": priority,
        "reason": f"{reason}: {sid}",
        "reason_json": {"trigger": action.value.lower(), "strategy": sid},
        "strategy_id": sid,
        "expected_benefit": benefit,
        "expected_risk": risk,
    }


def _maps_to_age_v2(action: str | None) -> str | None:
    mapping = {
        PredictiveAction.REBALANCE_EARLY.value: "ALLOCATION_REBALANCE",
        PredictiveAction.PREPARE_RECOVERY.value: "ENTER_RECOVERY",
        PredictiveAction.REDUCE_RISK.value: "REDUCE_RISK",
        PredictiveAction.PROFILE_SWITCH.value: "PROFILE_SWITCH",
        PredictiveAction.PROMOTE_STRATEGY.value: "PROMOTE_STRATEGY",
        PredictiveAction.DEMOTE_STRATEGY.value: "DEMOTE_STRATEGY",
        PredictiveAction.RETIRE_STRATEGY.value: "RETIRE_STRATEGY",
    }
    return mapping.get(str(action or ""))


def _priority_rank(priority: str) -> int:
    return {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}.get(priority.upper(), 0)
