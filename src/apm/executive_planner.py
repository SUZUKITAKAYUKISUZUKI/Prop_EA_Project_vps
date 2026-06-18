"""Executive action planning for APM v1."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.apm.executive_context import ExecutiveContext


class ExecutivePlanner:
    def plan(
        self,
        context: ExecutiveContext,
        *,
        policies: dict[str, Any],
        governance: dict[str, Any],
        opportunities: list[dict[str, Any]],
        risks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        confidence = float(context.confidence.get("portfolio_confidence") or 70)
        recommended = str(context.confidence.get("recommended_action") or "NO_ACTION")

        if policies.get("recovery_mode"):
            actions.append(self._action("ENTER_RECOVERY", context, confidence, expected_benefit=2.0, expected_risk=-6.0))

        for opp in opportunities[:2]:
            code = opp.get("strategy")
            actions.append(
                self._action(
                    "PROMOTE_STRATEGY",
                    context,
                    confidence,
                    strategy=code,
                    expected_benefit=float(opp.get("portfolio_fit") or 0) / 10.0,
                    expected_risk=-3.0,
                    details=opp,
                )
            )

        for risk in risks[:2]:
            code = risk.get("strategy")
            if float(risk.get("risk_score") or 0) >= 80:
                actions.append(
                    self._action(
                        "DEMOTE_STRATEGY",
                        context,
                        confidence,
                        strategy=code,
                        expected_benefit=1.0,
                        expected_risk=float(risk.get("health_impact") or -5.0),
                        details=risk,
                    )
                )

        current = context.allocation.get("current_weights") or {}
        recommended_weights = context.allocation.get("recommended_weights") or {}
        if recommended_weights and self._allocation_drift(current, recommended_weights) > 0.05:
            actions.append(
                self._action(
                    "ALLOCATION_REBALANCE",
                    context,
                    confidence,
                    expected_benefit=4.0,
                    expected_risk=-2.0,
                    details={"current": current, "recommended": recommended_weights},
                )
            )

        if "PROFILE_SWITCH" in recommended and policies.get("allow_profile_switch"):
            actions.append(
                self._action(
                    "PROFILE_SWITCH",
                    context,
                    confidence,
                    expected_benefit=12.0,
                    expected_risk=-4.0,
                )
            )

        if not actions:
            actions.append(self._action("NO_ACTION", context, confidence))
        return actions

    def _allocation_drift(self, current: dict[str, Any], recommended: dict[str, Any]) -> float:
        keys = set(current) | set(recommended)
        return sum(abs(float(recommended.get(k, 0)) - float(current.get(k, 0))) for k in keys)

    def _action(
        self,
        action_type: str,
        context: ExecutiveContext,
        confidence: float,
        *,
        strategy: str | None = None,
        expected_benefit: float = 0.0,
        expected_risk: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "action_id": str(uuid4()),
            "profile_id": context.profile_id,
            "action_type": action_type,
            "strategy": strategy,
            "confidence": round(confidence, 2),
            "expected_benefit_pct": round(expected_benefit, 2),
            "expected_risk_pct": round(expected_risk, 2),
            "status": "PENDING_APPROVAL",
            "details": details or {},
        }
