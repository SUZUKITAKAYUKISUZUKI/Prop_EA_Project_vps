"""Execute a single PDTS scenario simulation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.digital_twin.allocation_override import apply_allocation_override
from src.digital_twin.challenge_simulator import ChallengeSimulator
from src.digital_twin.funded_simulator import FundedSimulator
from src.digital_twin.monte_carlo_wrapper import MonteCarloWrapper
from src.digital_twin.portfolio_clone import clone_profile_context
from src.digital_twin.recovery_simulator import RecoverySimulator
from src.digital_twin.recommendation_engine import RecommendationEngine
from src.digital_twin.scenario_builder import ScenarioDefinition
from src.services.profile_service import ProfileContext


class ScenarioRunner:
    def __init__(
        self,
        *,
        challenge: ChallengeSimulator | None = None,
        funded: FundedSimulator | None = None,
        recovery: RecoverySimulator | None = None,
        monte_carlo: MonteCarloWrapper | None = None,
        recommender: RecommendationEngine | None = None,
    ) -> None:
        self._challenge = challenge or ChallengeSimulator()
        self._funded = funded or FundedSimulator()
        self._recovery = recovery or RecoverySimulator()
        self._mc = monte_carlo or MonteCarloWrapper()
        self._recommender = recommender or RecommendationEngine()

    def run(
        self,
        scenario: ScenarioDefinition,
        trades: pd.DataFrame,
        profile_ctx: ProfileContext,
        *,
        health_score: float = 50.0,
        mc_fast: bool = True,
        persist_mc: bool = True,
    ) -> dict[str, Any]:
        twin_ctx = clone_profile_context(profile_ctx, suffix=scenario.name)
        apply_allocation_override(twin_ctx, scenario.allocation)

        state = scenario.account_state.lower()
        if state == "challenge":
            metrics = self._challenge.run(
                trades,
                scenario.allocation,
                twin_ctx,
                health_score=health_score,
                fast=mc_fast,
            )
        elif state == "recovery":
            metrics = self._recovery.run(trades, scenario.allocation, health_score=health_score)
        else:
            metrics = self._funded.run(trades, scenario.allocation, health_score=health_score)

        monte_carlo = {}
        if persist_mc and not trades.empty:
            monte_carlo = self._mc.run_paths(
                trades,
                scenario.allocation,
                twin_ctx,
                account_state=state,
                fast=mc_fast,
            )
            primary = monte_carlo.get("500") or monte_carlo.get("100") or {}
            if primary:
                metrics["pass_rate"] = primary.get("pass_rate", metrics.get("pass_rate"))
                metrics["avg_pass_days"] = primary.get("expected_days", metrics.get("avg_pass_days"))

        rec = self._recommender.evaluate(metrics)
        allocation_pct = {k: round(v * 100.0, 1) for k, v in scenario.allocation.items() if v > 0}

        return {
            "scenario": scenario.name,
            "scenario_label": scenario.label,
            "profile_id": scenario.profile_id,
            "account_state": state,
            "allocation": allocation_pct,
            "score": rec["score"],
            "recommendation": rec["recommendation"],
            "pass_rate": metrics.get("pass_rate"),
            "avg_pass_days": metrics.get("avg_pass_days"),
            "pf": metrics.get("pf"),
            "total_r": metrics.get("total_r"),
            "max_dd": metrics.get("max_dd"),
            "sharpe": metrics.get("sharpe"),
            "win_rate": metrics.get("win_rate"),
            "recovery_factor": metrics.get("recovery_factor"),
            "ulcer_index": metrics.get("ulcer_index"),
            "risk_score": metrics.get("risk_score"),
            "health_score": metrics.get("health_score"),
            "prob_recovery": metrics.get("prob_recovery"),
            "prob_ruin": metrics.get("prob_ruin"),
            "monte_carlo": monte_carlo,
            "metrics": metrics,
        }
