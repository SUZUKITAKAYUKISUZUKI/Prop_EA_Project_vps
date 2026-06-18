"""Dashboard reporting for Strategy Lifecycle Manager."""

from __future__ import annotations



from typing import Any



from src.strategy_lifecycle.lifecycle_stages import CORE_MIN_ALLOCATION, STAGE_ALLOCATION





class LifecycleReport:

    @staticmethod

    def _fit_breakdown(metrics: dict[str, Any], reg: dict[str, Any]) -> dict[str, Any]:

        return {

            "diversification": metrics.get("diversification_score") or reg.get("diversification_score"),

            "recovery": metrics.get("recovery_score") or reg.get("recovery_score"),

            "challenge": metrics.get("challenge_score") or reg.get("challenge_score"),

            "stability": metrics.get("stability_contribution_score") or reg.get("stability_contribution_score"),

            "dd_reduction": metrics.get("dd_reduction_score") or reg.get("dd_reduction_score"),

        }



    def build_dashboard(

        self,

        *,

        registry: list[dict[str, Any]],

        evaluations: list[dict[str, Any]],

        history: list[dict[str, Any]] | None = None,

        fit_report: dict[str, Any] | None = None,

        fit_bundle: dict[str, dict[str, Any]] | None = None,

        genealogy: dict[str, Any] | None = None,

        decision_report: list[dict[str, Any]] | None = None,

    ) -> dict[str, Any]:

        eval_map = {e["strategy_id"]: e for e in evaluations}

        rows: list[dict[str, Any]] = []

        for reg in registry:

            sid = str(reg["strategy_id"])

            metrics = eval_map.get(sid, {})

            fit = (fit_bundle or {}).get(sid, {})

            stage = str(reg.get("current_stage") or "INCUBATION")

            fixed = STAGE_ALLOCATION.get(stage.upper())

            if stage.upper() == "CORE":

                allocation = f"min {round(CORE_MIN_ALLOCATION * 100.0, 0)}% + PAAE"

            elif fixed is None:

                allocation = "PAAE"

            else:

                allocation = f"{round(float(fixed) * 100.0, 1)}%"

            breakdown = self._fit_breakdown(metrics, reg)

            rows.append(

                {

                    "strategy": sid,

                    "strategy_name": reg.get("strategy_name") or sid,

                    "strategy_version": reg.get("strategy_version") or metrics.get("strategy_version"),

                    "stage": stage,

                    "score": metrics.get("score", reg.get("score")),

                    "portfolio_fit_score": metrics.get(

                        "portfolio_fit_score",

                        reg.get("portfolio_fit_score"),

                    ),

                    "portfolio_fit_breakdown": breakdown,

                    "pf": metrics.get("pf"),

                    "pass_rate": metrics.get("pass_rate"),

                    "allocation": allocation,

                    "eligible": metrics.get("eligible", False),

                    "candidate_readiness": metrics.get("candidate_readiness", False),

                    "next_stage": metrics.get("next_stage"),

                    "core": stage.upper() == "CORE" or int(reg.get("core_strategy") or 0) == 1,

                    "active": bool(reg.get("active", 1)),

                    "correlation": fit.get("average_correlation"),

                    "recovery_impact": fit.get("recovery_impact"),

                    "challenge_impact": fit.get("challenge_impact"),

                    "health_impact": fit.get("health_impact"),

                }

            )



        promotion_candidates = sorted(

            [

                r

                for r in rows

                if r.get("eligible") and r.get("next_stage") in {"PRODUCTION", "CORE", "CANDIDATE"}

            ],

            key=lambda r: float(r.get("portfolio_fit_score") or 0.0),

            reverse=True,

        )

        recovery_candidates = [r for r in rows if r.get("stage") == "RECOVERY"]

        retired = [r for r in rows if r.get("stage") == "RETIRED"]

        retirement_candidates = sorted(

            [r for r in rows if r.get("stage") in {"RECOVERY", "PRODUCTION", "CORE"}],

            key=lambda r: float(r.get("portfolio_fit_score") or 0.0),

        )

        core_strategies = [r for r in rows if r.get("core")]



        stage_counts: dict[str, int] = {}

        for row in rows:

            stage_counts[row["stage"]] = stage_counts.get(row["stage"], 0) + 1



        fit_report = fit_report or {}

        return {

            "strategies": rows,

            "promotion_candidates": promotion_candidates,

            "recovery_candidates": recovery_candidates,

            "retirement_candidates": retirement_candidates,

            "retired_strategies": retired,

            "core_strategies": core_strategies,

            "strategy_genealogy": genealogy or {},

            "lifecycle_decisions": decision_report or [],

            "stage_distribution": stage_counts,

            "portfolio_fit_ranking": fit_report.get("ranking") or [],

            "portfolio_fit_report": fit_report,

            "portfolio_fit_distribution": fit_report.get("fit_distribution") or {},

            "correlation_fit_scatter": [

                {

                    "strategy": sid,

                    "correlation": data.get("average_correlation"),

                    "portfolio_fit_score": data.get("portfolio_fit_score"),

                }

                for sid, data in (fit_bundle or {}).items()

            ],

            "history": history or [],

        }



    def evaluate_response(self, evaluation: dict[str, Any]) -> dict[str, Any]:

        return {

            "strategy": evaluation.get("strategy_id"),

            "stage": evaluation.get("current_stage"),

            "strategy_version": evaluation.get("strategy_version"),

            "score": evaluation.get("score"),

            "portfolio_fit_score": evaluation.get("portfolio_fit_score"),

            "portfolio_fit_breakdown": {

                "diversification": evaluation.get("diversification_score"),

                "recovery": evaluation.get("recovery_score"),

                "challenge": evaluation.get("challenge_score"),

                "stability": evaluation.get("stability_contribution_score"),

                "dd_reduction": evaluation.get("dd_reduction_score"),

            },

            "candidate_readiness": evaluation.get("candidate_readiness"),

            "next_stage": evaluation.get("next_stage"),

            "eligible": evaluation.get("eligible", False),

            "reason": evaluation.get("reason"),

            "explanation": (evaluation.get("explanation") or {}).get("reason"),

            "pf": evaluation.get("pf"),

            "pass_rate": evaluation.get("pass_rate"),

            "allocation": evaluation.get("allocation"),

            "core": bool(evaluation.get("core_strategy") or evaluation.get("current_stage") == "CORE"),

        }

