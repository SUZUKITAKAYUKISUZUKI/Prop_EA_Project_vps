"""Strategy Lifecycle Manager orchestrator."""

from __future__ import annotations



from typing import Any



import pandas as pd



from src.digital_twin.scenario_repository import ScenarioRepository

from src.strategy_lifecycle.demotion_engine import DemotionEngine

from src.strategy_lifecycle.explainability_engine import ExplainabilityEngine

from src.strategy_lifecycle.genealogy_repository import GenealogyRepository

from src.strategy_lifecycle.incubation_engine import IncubationEngine

from src.strategy_lifecycle.lifecycle_manager import LifecycleManager

from src.strategy_lifecycle.lifecycle_repository import LifecycleRepository

from src.strategy_lifecycle.lifecycle_report import LifecycleReport

from src.strategy_lifecycle.portfolio_fit_engine import PortfolioFitEngine

from src.strategy_lifecycle.portfolio_fit_report import PortfolioFitReport

from src.strategy_lifecycle.promotion_engine import PromotionEngine

from src.strategy_lifecycle.qualification_engine import QualificationEngine

from src.strategy_lifecycle.retirement_engine import RetirementEngine





class StrategyLifecycleEngine:

    EVALUATION_INTERVAL_DAYS = 7



    def __init__(

        self,

        *,

        repo: LifecycleRepository | None = None,

        qualification: QualificationEngine | None = None,

        promotion: PromotionEngine | None = None,

        demotion: DemotionEngine | None = None,

        retirement: RetirementEngine | None = None,

        incubation: IncubationEngine | None = None,

        manager: LifecycleManager | None = None,

        reporter: LifecycleReport | None = None,

        portfolio_fit: PortfolioFitEngine | None = None,

        fit_report: PortfolioFitReport | None = None,

        explainability: ExplainabilityEngine | None = None,

        genealogy: GenealogyRepository | None = None,

        scenarios: ScenarioRepository | None = None,

        owns_connections: bool = False,

    ) -> None:

        self._owns = owns_connections or repo is None

        self._repo = repo or LifecycleRepository(owns_connection=self._owns)

        self._qualification = qualification or QualificationEngine()

        self._promotion = promotion or PromotionEngine()

        self._demotion = demotion or DemotionEngine()

        self._retirement = retirement or RetirementEngine()

        self._incubation = incubation or IncubationEngine()

        self._manager = manager or LifecycleManager()

        self._reporter = reporter or LifecycleReport()

        self._portfolio_fit = portfolio_fit or PortfolioFitEngine()

        self._fit_report = fit_report or PortfolioFitReport()

        self._explain = explainability or ExplainabilityEngine()

        self._genealogy = genealogy or GenealogyRepository(owns_connection=False)

        self._scenarios = scenarios or ScenarioRepository(owns_connection=False)



    def close(self) -> None:

        if self._owns:

            self._repo.close()

        self._qualification.close()

        self._scenarios.close()

        self._genealogy.close()



    def _pdts_score(self, profile_id: str) -> float | None:

        latest = self._scenarios.get_latest(profile_id, "recommended")

        if not latest:

            return None

        return float(latest.get("recommendation_score") or 0.0)



    def _build_fit_bundle(

        self,

        *,

        prae_v2: dict[str, Any] | None,

        state_summary: dict[str, Any] | None,

        trades: pd.DataFrame | None,

    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:

        all_metrics = self._qualification.build_all_base_metrics(

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

        )

        fit_bundle = self._portfolio_fit.compute_all(all_metrics, trades)

        return all_metrics, fit_bundle



    @staticmethod

    def _enrich_from_registry(metrics: dict[str, Any], reg: dict[str, Any]) -> dict[str, Any]:

        return {

            **metrics,

            "strategy_id": metrics.get("strategy_id") or reg.get("strategy_id"),

            "strategy_version": reg.get("strategy_version") or metrics.get("strategy_version") or "1.0",

            "core_strategy": int(reg.get("core_strategy") or 0),

            "current_stage": reg.get("current_stage") or metrics.get("current_stage"),

        }



    @staticmethod

    def _transition_fields(evaluation: dict[str, Any], reg: dict[str, Any] | None = None) -> dict[str, Any]:

        reg = reg or {}

        return {

            "score": evaluation.get("score"),

            "portfolio_fit_score": evaluation.get("portfolio_fit_score"),

            "pf": evaluation.get("pf"),

            "pass_rate": evaluation.get("pass_rate"),

            "max_dd": evaluation.get("max_dd"),

            "oos_pf": evaluation.get("oos_pf"),

            "strategy_version": evaluation.get("strategy_version") or reg.get("strategy_version"),

            "diversification_score": evaluation.get("diversification_score"),

            "recovery_score": evaluation.get("recovery_score"),

            "challenge_score": evaluation.get("challenge_score"),

            "stability_contribution_score": evaluation.get("stability_contribution_score"),

            "dd_reduction_score": evaluation.get("dd_reduction_score"),

        }

    @staticmethod
    def _registry_update_fields(evaluation: dict[str, Any], reg: dict[str, Any] | None = None) -> dict[str, Any]:
        reg = reg or {}
        return {
            "score": evaluation.get("score"),
            "portfolio_fit_score": evaluation.get("portfolio_fit_score"),
            "strategy_version": evaluation.get("strategy_version") or reg.get("strategy_version"),
            "diversification_score": evaluation.get("diversification_score"),
            "recovery_score": evaluation.get("recovery_score"),
            "challenge_score": evaluation.get("challenge_score"),
            "stability_contribution_score": evaluation.get("stability_contribution_score"),
            "dd_reduction_score": evaluation.get("dd_reduction_score"),
        }

    def get_portfolio_fit_ranking(

        self,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

    ) -> dict[str, Any]:

        _, fit_bundle = self._build_fit_bundle(

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

        )

        return self._fit_report.build(fit_bundle)



    def get_strategy_portfolio_fit(

        self,

        strategy_id: str,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

    ) -> dict[str, Any]:

        _, fit_bundle = self._build_fit_bundle(

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

        )

        return fit_bundle.get(strategy_id) or self._portfolio_fit.compute(strategy_id, {})



    def get_portfolio_fit_score(

        self,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

    ) -> dict[str, Any]:

        _, fit_bundle = self._build_fit_bundle(

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

        )

        report = self._fit_report.build(fit_bundle)

        return {

            "average_fit": report.get("average_fit"),

            "highest_fit_strategy": report.get("highest_fit_strategy"),

            "lowest_fit_strategy": report.get("lowest_fit_strategy"),

            "fit_distribution": report.get("fit_distribution"),

        }



    def evaluate_strategy(

        self,

        strategy_id: str,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

        profile_id: str | None = None,

        fit_bundle: dict[str, dict[str, Any]] | None = None,

    ) -> dict[str, Any]:

        reg = self._repo.get_strategy(strategy_id)

        if not reg:

            reg = self._repo.register_strategy(strategy_id)

        current = str(reg.get("current_stage") or "INCUBATION")

        metrics = self._qualification.build_metrics(

            strategy_id,

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

            fit_bundle=fit_bundle,

        )

        metrics = self._enrich_from_registry(metrics, reg)

        pdts_score = self._pdts_score(profile_id or "ChallengeAggressive") if profile_id else None



        next_stage, reason, eligible = self._promotion.next_stage(current, metrics, pdts_score=pdts_score)

        if not eligible:

            demote_stage, demote_reason, demote_ok = self._demotion.next_stage(current, metrics)

            if demote_ok and demote_stage:

                next_stage, reason, eligible = demote_stage, demote_reason, False

            else:

                retire_stage, retire_reason, retire_ok = self._retirement.next_stage(

                    current,

                    metrics,

                    recovery_started_at=reg.get("demoted_at"),

                    strategy_version=str(reg.get("strategy_version") or "1.0"),

                    previous_version=str(reg.get("strategy_version") or "1.0"),

                )

                if retire_ok and retire_stage:

                    next_stage, reason, eligible = retire_stage, retire_reason, False



        stage = current

        fixed = self._manager.allocation_for_stage(stage)

        allocation = "PAAE" if fixed is None else f"{round(float(fixed or 0) * 100.0, 1)}%"

        if stage == "CORE":

            allocation = f"min {round(self._manager.core_min_weights([reg]).get(strategy_id, 0.10) * 100.0, 0)}% + PAAE"



        self._repo.save_evaluation(strategy_id, {**metrics, "current_stage": current})



        return {

            **metrics,

            "current_stage": current,

            "next_stage": next_stage,

            "eligible": eligible and next_stage is not None,

            "reason": reason,

            "allocation": allocation,

            "pdts_score": pdts_score,

        }



    def promote_strategy(

        self,

        strategy_id: str,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

        profile_id: str | None = None,

        force: bool = False,

    ) -> dict[str, Any]:

        evaluation = self.evaluate_strategy(

            strategy_id,

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

            profile_id=profile_id,

        )

        if not force and not evaluation.get("eligible"):

            return {**evaluation, "promoted": False}



        reg = self._repo.get_strategy(strategy_id) or {}

        old_stage = str(reg.get("current_stage") or evaluation["current_stage"])

        new_stage = str(evaluation.get("next_stage") or old_stage)

        if new_stage == old_stage and not force:

            return {**evaluation, "promoted": False}



        if force and evaluation.get("next_stage") is None:

            transitions = {

                "INCUBATION": "CANDIDATE",

                "CANDIDATE": "PRODUCTION",

                "PRODUCTION": "CORE",

                "RECOVERY": "PRODUCTION",

            }

            new_stage = transitions.get(old_stage, old_stage)



        from src.database.db_manager import utc_now_iso



        now = utc_now_iso()

        ts_kwargs = self._registry_update_fields(evaluation, reg)

        if new_stage == "PRODUCTION":

            ts_kwargs["promoted_at"] = now

        elif new_stage == "CORE":

            ts_kwargs["promoted_at"] = now

            ts_kwargs["core_strategy"] = 1

        elif new_stage == "RECOVERY":

            ts_kwargs["demoted_at"] = now

            ts_kwargs["core_strategy"] = 0

        elif new_stage == "RETIRED":

            ts_kwargs["retired_at"] = now

            ts_kwargs["core_strategy"] = 0



        self._repo.update_stage(strategy_id, new_stage, **ts_kwargs)

        self._repo.log_transition(

            strategy_id=strategy_id,

            old_stage=old_stage,

            new_stage=new_stage,

            reason=str(evaluation.get("reason") or "manual_promote"),

            **self._transition_fields(evaluation, reg),

        )

        if new_stage == "CORE":

            version = str(evaluation.get("strategy_version") or reg.get("strategy_version") or "1.0")

            self._genealogy.add_version(strategy_id, version, parent_strategy_id=strategy_id)



        evaluation["current_stage"] = new_stage

        evaluation["core_strategy"] = 1 if new_stage == "CORE" else 0

        evaluation["promoted"] = new_stage != old_stage

        evaluation["explanation"] = self._explain.explain(

            {**evaluation, "next_stage": new_stage},

            decision="PROMOTE" if new_stage != old_stage else "HOLD",

        )

        return evaluation



    def retire_strategy(self, strategy_id: str, *, reason: str = "manual_retire") -> dict[str, Any]:

        reg = self._repo.get_strategy(strategy_id)

        if not reg:

            raise KeyError(f"Unknown strategy: {strategy_id}")

        old_stage = str(reg.get("current_stage"))

        from src.database.db_manager import utc_now_iso



        self._repo.update_stage(strategy_id, "RETIRED", retired_at=utc_now_iso(), core_strategy=0)

        self._repo.log_transition(

            strategy_id=strategy_id,

            old_stage=old_stage,

            new_stage="RETIRED",

            reason=reason,

            score=reg.get("score"),

            portfolio_fit_score=reg.get("portfolio_fit_score"),

            strategy_version=reg.get("strategy_version"),

        )

        return {"strategy_id": strategy_id, "old_stage": old_stage, "current_stage": "RETIRED", "retired": True}



    def run_weekly_evaluation(

        self,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

        profile_id: str | None = None,

        force: bool = False,

        auto_apply: bool = True,

    ) -> dict[str, Any]:

        days = self._repo.days_since_last_evaluation()

        if not force and days is not None and days < self.EVALUATION_INTERVAL_DAYS:

            return {"evaluated": False, "reason": "within_weekly_cooldown", "days_since_last": days}



        registry = self._repo.list_strategies()

        _, fit_bundle = self._build_fit_bundle(

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

        )

        evaluations: list[dict[str, Any]] = []

        transitions: list[dict[str, Any]] = []



        for reg in registry:

            if not reg.get("active", 1):

                continue

            sid = str(reg["strategy_id"])

            ev = self.evaluate_strategy(

                sid,

                prae_v2=prae_v2,

                state_summary=state_summary,

                trades=trades,

                profile_id=profile_id,

                fit_bundle=fit_bundle,

            )

            evaluations.append(ev)

            if auto_apply and ev.get("eligible") and ev.get("next_stage"):

                result = self.promote_strategy(

                    sid,

                    prae_v2=prae_v2,

                    state_summary=state_summary,

                    trades=trades,

                    profile_id=profile_id,

                )

                if result.get("promoted"):

                    transitions.append(result)

            elif auto_apply:

                demote_stage, demote_reason, demote_ok = self._demotion.next_stage(

                    str(reg.get("current_stage")), ev

                )

                if demote_ok and demote_stage:

                    from src.database.db_manager import utc_now_iso



                    demote_kwargs = self._registry_update_fields(ev, reg)

                    demote_kwargs["demoted_at"] = utc_now_iso()

                    demote_kwargs["core_strategy"] = 0

                    self._repo.update_stage(sid, demote_stage, **demote_kwargs)

                    self._repo.log_transition(

                        strategy_id=sid,

                        old_stage=str(reg.get("current_stage")),

                        new_stage=demote_stage,

                        reason=f"weekly_evaluation:{demote_reason}",

                        **self._transition_fields(ev, reg),

                    )

                    transitions.append({**ev, "current_stage": demote_stage, "promoted": False, "demoted": True})

                else:

                    retire_stage, retire_reason, retire_ok = self._retirement.next_stage(

                        str(reg.get("current_stage")),

                        ev,

                        recovery_started_at=reg.get("demoted_at"),

                    )

                    if retire_ok and retire_stage:

                        self.retire_strategy(sid, reason=f"weekly_evaluation:{retire_reason}")

                        transitions.append({**ev, "current_stage": retire_stage, "retired": True})



        self._repo.log_transition(

            strategy_id="PORTFOLIO",

            old_stage=None,

            new_stage="EVALUATED",

            reason="weekly_evaluation_complete",

        )

        report = self._reporter.build_dashboard(

            registry=self._repo.list_strategies(),

            evaluations=evaluations,

            history=self._repo.list_history(limit=20),

            fit_bundle=fit_bundle,

            genealogy=self._genealogy.build_tree(),

            decision_report=self._explain.decision_report(evaluations),

        )

        return {

            "evaluated": True,

            "transitions": transitions,

            "evaluations": evaluations,

            **report,

        }



    def get_lifecycle_dashboard(

        self,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

        profile_id: str | None = None,

    ) -> dict[str, Any]:

        registry = self._repo.list_strategies()

        _, fit_bundle = self._build_fit_bundle(

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

        )

        evaluations = [

            self.evaluate_strategy(

                str(r["strategy_id"]),

                prae_v2=prae_v2,

                state_summary=state_summary,

                trades=trades,

                profile_id=profile_id,

                fit_bundle=fit_bundle,

            )

            for r in registry

            if r.get("active", 1)

        ]

        fit_report = self._fit_report.build(fit_bundle)

        return self._reporter.build_dashboard(

            registry=registry,

            evaluations=evaluations,

            history=self._repo.list_history(limit=20),

            fit_report=fit_report,

            fit_bundle=fit_bundle,

            genealogy=self._genealogy.build_tree(),

            decision_report=self._explain.decision_report(evaluations),

        )



    def get_strategy_genealogy(self, strategy_id: str | None = None) -> dict[str, Any]:

        return self._genealogy.build_tree(strategy_id)



    def get_strategy_explanation(

        self,

        strategy_id: str,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

        profile_id: str | None = None,

    ) -> dict[str, Any]:

        evaluation = self.evaluate_strategy(

            strategy_id,

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

            profile_id=profile_id,

        )

        decision = "PROMOTE" if evaluation.get("eligible") else "HOLD"

        if evaluation.get("reason", "").startswith("score_below") or "portfolio_fit_below" in str(

            evaluation.get("reason")

        ):

            if evaluation.get("current_stage") in {"PRODUCTION", "CORE"}:

                decision = "DEMOTE"

        if evaluation.get("next_stage") == "RETIRED":

            decision = "RETIRE"

        return self._explain.explain(evaluation, decision=decision)



    def get_core_strategies(self) -> list[dict[str, Any]]:

        registry = self._repo.list_strategies()

        core_ids = self._manager.core_strategies(registry)

        rows: list[dict[str, Any]] = []

        for reg in registry:

            sid = str(reg["strategy_id"])

            if sid not in core_ids:

                continue

            rows.append(

                {

                    "strategy": sid,

                    "strategy_version": reg.get("strategy_version"),

                    "stage": reg.get("current_stage"),

                    "score": reg.get("score"),

                    "portfolio_fit_score": reg.get("portfolio_fit_score"),

                    "min_allocation_pct": round(self._manager.core_min_weights([reg]).get(sid, 0.10) * 100.0, 1),

                    "diversification_score": reg.get("diversification_score"),

                    "recovery_score": reg.get("recovery_score"),

                    "challenge_score": reg.get("challenge_score"),

                    "stability_contribution_score": reg.get("stability_contribution_score"),

                    "dd_reduction_score": reg.get("dd_reduction_score"),

                }

            )

        return rows



    def get_lifecycle_decision_report(

        self,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

        profile_id: str | None = None,

    ) -> list[dict[str, Any]]:

        registry = self._repo.list_strategies()

        _, fit_bundle = self._build_fit_bundle(

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

        )

        evaluations = [

            self.evaluate_strategy(

                str(r["strategy_id"]),

                prae_v2=prae_v2,

                state_summary=state_summary,

                trades=trades,

                profile_id=profile_id,

                fit_bundle=fit_bundle,

            )

            for r in registry

            if r.get("active", 1)

        ]

        return self._explain.decision_report(evaluations)



    def get_strategy_governance_context(

        self,

        strategy_id: str,

        *,

        prae_v2: dict[str, Any] | None = None,

        state_summary: dict[str, Any] | None = None,

        trades: pd.DataFrame | None = None,

        profile_id: str | None = None,

    ) -> dict[str, Any]:

        evaluation = self.evaluate_strategy(

            strategy_id,

            prae_v2=prae_v2,

            state_summary=state_summary,

            trades=trades,

            profile_id=profile_id,

        )

        return self._explain.governance_context(evaluation)



    def apply_lifecycle_to_weights(

        self,

        weights: dict[str, float],

        *,

        registry: list[dict[str, Any]] | None = None,

    ) -> tuple[dict[str, float], dict[str, str]]:

        reg = registry or self._repo.list_strategies()

        stages = self._manager.stage_map(reg)

        return self._manager.apply_stage_allocations(weights, stages, registry=reg)

