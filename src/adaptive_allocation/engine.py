"""Portfolio Adaptive Allocation Engine orchestrator."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.adaptive_allocation.allocation_constraints import AllocationConstraints, validate_weights
from src.adaptive_allocation.allocation_history import AllocationHistoryRepository
from src.adaptive_allocation.allocation_optimizer import AllocationOptimizer
from src.adaptive_allocation.allocation_policy import policy_metadata, resolve_account_state
from src.adaptive_allocation.allocation_report import AllocationReport
from src.adaptive_allocation.allocation_simulator import AllocationSimulator
from src.adaptive_allocation.profile_weight_writer import ProfileWeightWriter
from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES
from src.strategy_lifecycle.lifecycle_manager import LifecycleManager
from src.strategy_lifecycle.lifecycle_repository import LifecycleRepository
from src.strategy_lifecycle.qualification_engine import QualificationEngine


class AdaptiveAllocationEngine:
    """Combine PRAE v2, State Analytics, and Profile Manager for adaptive weights."""

    def __init__(
        self,
        *,
        history: AllocationHistoryRepository | None = None,
        optimizer: AllocationOptimizer | None = None,
        simulator: AllocationSimulator | None = None,
        reporter: AllocationReport | None = None,
        writer: ProfileWeightWriter | None = None,
        constraints: AllocationConstraints | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections or history is None
        self._history = history or AllocationHistoryRepository(owns_connection=self._owns)
        self._optimizer = optimizer or AllocationOptimizer()
        self._simulator = simulator or AllocationSimulator()
        self._reporter = reporter or AllocationReport(constraints)
        self._writer = writer or ProfileWeightWriter(owns_connection=False)
        self._constraints = constraints or AllocationConstraints()

    def close(self) -> None:
        if self._owns:
            self._history.close()

    def should_rebalance(
        self,
        profile_id: str,
        *,
        force: bool = False,
        trigger: str | None = None,
    ) -> tuple[bool, str]:
        if force:
            return True, trigger or "forced"
        if trigger in {"recovery_end", "funded_transition", "challenge_passed"}:
            return True, trigger

        days = self._history.days_since_last_rebalance(profile_id)
        if days is None:
            return True, "initial_rebalance"
        if days >= self._constraints.rebalance_interval_days:
            return True, "weekly_schedule"
        return False, "within_cooldown"

    def compute_allocation(
        self,
        *,
        prae_v2: dict[str, Any],
        current_weights: dict[str, float],
        profile_id: str,
        profile_settings: dict[str, str] | None = None,
        state_summary: dict[str, Any] | None = None,
        trades: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        state_summary = state_summary or {}
        account_state = str(
            state_summary.get("current_state") or resolve_account_state(profile_id, profile_settings)
        ).lower()
        policy = policy_metadata(account_state)

        inputs = self._optimizer.build_strategy_inputs(prae_v2)
        lifecycle_repo = LifecycleRepository(owns_connection=False)
        registry = lifecycle_repo.list_strategies()
        lifecycle_manager = LifecycleManager()
        lifecycle_stages = lifecycle_manager.stage_map(registry)
        core_min_weights = lifecycle_manager.core_min_weights(registry)
        qualification = QualificationEngine()
        portfolio_fit_scores: dict[str, float] = {}
        try:
            all_metrics = qualification.build_all_base_metrics(
                prae_v2=prae_v2,
                state_summary=state_summary,
                trades=trades,
            )
            fit_bundle = qualification._fit.compute_all(all_metrics, trades)
            portfolio_fit_scores = {
                sid: float(data.get("portfolio_fit_score") or 50.0)
                for sid, data in fit_bundle.items()
            }
        finally:
            qualification.close()
            lifecycle_repo.close()

        quality_scores = self._optimizer.compute_quality_scores(
            inputs,
            portfolio_fit_scores=portfolio_fit_scores,
        )
        recommended, disabled, reasons = self._optimizer.adjust_weights(
            current_weights,
            quality_scores,
            account_state=account_state,
            constraints=self._constraints,
            lifecycle_stages=lifecycle_stages,
            core_min_weights=core_min_weights,
        )
        recommended, lifecycle_reasons = lifecycle_manager.apply_stage_allocations(
            recommended,
            lifecycle_stages,
            registry=registry,
        )
        reasons.update(lifecycle_reasons)

        drift_alerts = self._reporter.detect_drift(current_weights, recommended)
        last_rebalance = self._history.last_rebalance_timestamp(profile_id)
        health = (prae_v2.get("health_report") or {}) | {
            "health_score": state_summary.get("health_score") or (prae_v2.get("health_report") or {}).get("health_score"),
        }

        simulation: dict[str, Any] = {}
        if trades is not None and not trades.empty:
            candidates = {
                "current": current_weights,
                "recommended": recommended,
                "policy_base": policy["base_weights"],
            }
            ranked = self._simulator.compare_candidates(trades, candidates, account_state=account_state)
            simulation = {"candidates": ranked, "best": ranked[0] if ranked else {}}

        report = self._reporter.build_report(
            current_weights=current_weights,
            recommended_weights=recommended,
            quality_scores=quality_scores,
            strategy_risk=prae_v2.get("strategy_risk") or [],
            health_report=health,
            drift_alerts=drift_alerts,
            last_rebalance=last_rebalance,
            simulation=simulation,
        )
        report["profile_id"] = profile_id
        report["account_state"] = account_state
        report["policy"] = policy
        report["strategy_inputs"] = inputs
        report["adjustment_reasons"] = reasons
        report["disabled_candidates"] = sorted(disabled)
        report["lifecycle_stages"] = lifecycle_stages
        report["portfolio_fit_scores"] = portfolio_fit_scores
        report["validation_errors"] = validate_weights(recommended, self._constraints)
        return report

    def rebalance(
        self,
        *,
        profile_id: str,
        prae_v2: dict[str, Any],
        current_weights: dict[str, float],
        profile_settings: dict[str, str] | None = None,
        state_summary: dict[str, Any] | None = None,
        trades: pd.DataFrame | None = None,
        apply: bool = True,
        force: bool = False,
        trigger: str | None = None,
    ) -> dict[str, Any]:
        allowed, reason = self.should_rebalance(profile_id, force=force, trigger=trigger)
        report = self.compute_allocation(
            prae_v2=prae_v2,
            current_weights=current_weights,
            profile_id=profile_id,
            profile_settings=profile_settings,
            state_summary=state_summary,
            trades=trades,
        )

        if not allowed:
            report["rebalanced"] = False
            report["rebalance_blocked_reason"] = reason
            return report

        recommended = {
            k: float(v) / 100.0
            for k, v in (report.get("recommended_weights") or {}).items()
        }
        for code in DASHBOARD_STRATEGY_CODES:
            recommended.setdefault(code, current_weights.get(code, 0.0))

        health_score = float((report.get("health_impact") or {}).get("health_score") or 0.0)
        inputs = report.get("strategy_inputs") or {}

        if apply:
            self._writer.apply_weights(profile_id, recommended, activate=True)
            for code in DASHBOARD_STRATEGY_CODES:
                old = float(current_weights.get(code, 0.0))
                new = float(recommended.get(code, 0.0))
                if abs(old - new) < 0.0001:
                    continue
                metrics = inputs.get(code, {})
                self._history.log_change(
                    profile_id=profile_id,
                    strategy=code,
                    old_weight=old,
                    new_weight=new,
                    reason=report.get("adjustment_reasons", {}).get(code, reason),
                    risk_score=float(metrics.get("risk_score") or 0.0),
                    profit_score=float(metrics.get("profit_contribution") or 0.0),
                    health_score=health_score,
                )

        report["rebalanced"] = apply
        report["rebalance_reason"] = reason
        report["last_rebalance"] = self._history.last_rebalance_timestamp(profile_id)
        return report
