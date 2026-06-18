"""Opportunity scoring from PDTS, AGE, PAAE."""
from __future__ import annotations

from typing import Any


class OpportunityEngine:
    def evaluate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        pdts = bundle.get("pdts") or {}
        age = bundle.get("age_v4") or {}
        paae = bundle.get("paae") or {}
        slm = bundle.get("slm") or {}

        cmp = pdts.get("scenario_comparison") or {}
        recommended = cmp.get("recommended") or {}
        metrics = age.get("best_future_metrics") or {}

        expected_r = float(metrics.get("expected_r") or recommended.get("expected_r") or 150)
        expected_pf = float(metrics.get("expected_pf") or 2.5)
        pass_probability = float(metrics.get("pass_probability") or recommended.get("pass_rate") or 90)
        future_health = float(metrics.get("health_score") or recommended.get("score") or 80)
        growth_potential = float(recommended.get("portfolio_fit_gain") or 10)

        opportunity_score = round(
            min(100.0, expected_r / 2.0) * 0.2
            + min(100.0, expected_pf * 30.0) * 0.15
            + pass_probability * 0.25
            + future_health * 0.25
            + min(100.0, growth_potential * 4.0) * 0.15,
            2,
        )

        top_opportunity = self._top_opportunity(paae, slm, age)
        opportunities = self._list_opportunities(bundle, opportunity_score)

        return {
            "opportunity_score": opportunity_score,
            "expected_r": expected_r,
            "expected_pf": expected_pf,
            "pass_probability": pass_probability,
            "future_health": future_health,
            "growth_potential": growth_potential,
            "top_opportunity": top_opportunity,
            "opportunities": opportunities,
        }

    def _top_opportunity(self, paae: dict[str, Any], slm: dict[str, Any], age: dict[str, Any]) -> str:
        for row in slm.get("promotion_candidates") or []:
            code = str(row.get("strategy") or "")
            if code:
                return f"Promote strategy {code}"
        current = paae.get("current_weights") or {}
        recommended = paae.get("recommended_weights") or {}
        for code, rec in recommended.items():
            cur = float(current.get(code, 0))
            rec_v = float(rec)
            if rec_v > cur + 0.03:
                return f"Increase allocation to {code}"
        action = str(age.get("recommended_action") or "")
        if action and action != "DO_NOTHING":
            return action.replace("_", " ").title()
        return "Maintain balanced growth posture"

    def _list_opportunities(self, bundle: dict[str, Any], score: float) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for row in (bundle.get("slm") or {}).get("promotion_candidates") or []:
            items.append(
                {
                    "strategy": row.get("strategy"),
                    "score": float(row.get("portfolio_fit_score") or row.get("score") or 0),
                    "source": "SLM",
                }
            )
        for row in (bundle.get("apm_v1") or {}).get("opportunities") or []:
            items.append({"strategy": row.get("strategy"), "score": float(row.get("portfolio_fit") or 0), "source": "APM"})
        if not items:
            items.append({"strategy": "PORTFOLIO", "score": score, "source": "CIL"})
        return items
