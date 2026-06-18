"""Explainability for SLM v3 lifecycle decisions."""
from __future__ import annotations

from typing import Any


class ExplainabilityEngine:
    """Generate human-readable promotion/demotion/retirement reasons."""

    def explain(
        self,
        evaluation: dict[str, Any],
        *,
        decision: str = "EVALUATE",
    ) -> dict[str, Any]:
        reasons: list[str] = []
        fit = float(evaluation.get("portfolio_fit_score") or 0.0)
        score = float(evaluation.get("score") or 0.0)
        pass_rate = float(evaluation.get("pass_rate") or 0.0)
        pf = float(evaluation.get("pf") or 0.0)
        oos_pf = float(evaluation.get("oos_pf") or 0.0)
        oos_months = float(evaluation.get("oos_months") or 0.0)
        div = evaluation.get("diversification_score")
        recovery_events = int(evaluation.get("recovery_events") or 0)
        next_stage = evaluation.get("next_stage")
        current = evaluation.get("current_stage")
        reason_code = str(evaluation.get("reason") or "")

        if decision in {"PROMOTE", "EVALUATE"} and evaluation.get("eligible"):
            if fit >= 60:
                reasons.append(f"Portfolio Fit {fit:.1f} >= 60")
            if pass_rate >= 95:
                reasons.append(f"Pass Rate {pass_rate:.1f}%")
            if pf >= 1.4:
                reasons.append(f"PF {pf:.2f}")
            if oos_pf >= 1.3:
                reasons.append(f"OOS PF {oos_pf:.2f}")
            if div is not None:
                reasons.append(f"Diversification Score {float(div):.1f}")
            if next_stage == "CORE":
                reasons.append(f"Score {score:.1f} >= 85")
                reasons.append(f"OOS period {oos_months:.1f} months >= 36")
                reasons.append(f"Recovery Events {recovery_events} <= 1")

        if decision in {"DEMOTE", "EVALUATE"} and reason_code:
            if "portfolio_fit_below" in reason_code:
                reasons.append(f"Portfolio Fit {fit:.1f} below threshold")
            if "score_below" in reason_code:
                reasons.append(f"Score {score:.1f} below threshold")
            if "dd_contribution" in reason_code:
                reasons.append("DD contribution elevated")
            if "recovery_events" in reason_code:
                reasons.append(f"Recovery events increased ({recovery_events})")
            if "prae_risk" in reason_code:
                reasons.append("PRAE risk warning triggered")

        if decision == "RETIRE" or next_stage == "RETIRED":
            if pf < 1.0:
                reasons.append(f"PF {pf:.2f} < 1.0")
            if fit < 20:
                reasons.append(f"Portfolio Fit {fit:.1f} < 20")

        if not reasons:
            reasons.append(str(reason_code or "no_transition_required"))

        return {
            "strategy": evaluation.get("strategy_id"),
            "decision": decision,
            "current_stage": current,
            "next_stage": next_stage,
            "reason": reasons,
            "reason_code": reason_code,
        }

    def decision_report(self, evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ev in evaluations:
            decision = "PROMOTE" if ev.get("eligible") else "HOLD"
            if ev.get("reason", "").startswith("score_below") or "portfolio_fit_below" in str(ev.get("reason")):
                if ev.get("current_stage") in {"PRODUCTION", "CORE"}:
                    decision = "DEMOTE"
            if ev.get("next_stage") == "RETIRED":
                decision = "RETIRE"
            rows.append(self.explain(ev, decision=decision))
        return rows

    def governance_context(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        explanation = self.explain(evaluation, decision="EVALUATE")
        return {
            "strategy": evaluation.get("strategy_id"),
            "stage": evaluation.get("current_stage"),
            "strategy_version": evaluation.get("strategy_version"),
            "score": evaluation.get("score"),
            "portfolio_fit": evaluation.get("portfolio_fit_score"),
            "core": bool(evaluation.get("core_strategy") or evaluation.get("current_stage") == "CORE"),
            "explanation": explanation.get("reason") or [],
            "portfolio_fit_breakdown": {
                "diversification": evaluation.get("diversification_score"),
                "recovery": evaluation.get("recovery_score"),
                "challenge": evaluation.get("challenge_score"),
                "stability": evaluation.get("stability_contribution_score"),
                "dd_reduction": evaluation.get("dd_reduction_score"),
            },
        }
