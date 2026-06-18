"""Challenge-phase scenario simulator."""
from __future__ import annotations

from typing import Any

import pandas as pd

from audit.risk_manager import STARTING_EQUITY
from core.pass_probability import (
    AccountSnapshot,
    ChallengeState,
    estimate_expected_pass_days,
    estimate_pass_probability,
)
from src.digital_twin._metrics_common import compute_scenario_metrics, prepare_weighted_trades
from src.services.profile_service import ProfileContext


class ChallengeSimulator:
    def run(
        self,
        trades: pd.DataFrame,
        weights: dict[str, float],
        profile_ctx: ProfileContext,
        *,
        health_score: float = 50.0,
        mc_trials: int = 500,
        fast: bool = True,
    ) -> dict[str, Any]:
        metrics = compute_scenario_metrics(trades, weights, health_score=health_score)
        weighted = prepare_weighted_trades(trades, weights)
        prop = profile_ctx.to_prop_profile()

        account = AccountSnapshot(
            equity=STARTING_EQUITY,
            balance=STARTING_EQUITY,
            peak_equity=STARTING_EQUITY,
            phase_start_equity=STARTING_EQUITY,
        )
        challenge = ChallengeState(
            days_elapsed=0,
            profit_progress_percent=0.0,
            daily_dd_used_percent=0.0,
            total_dd_used_percent=0.0,
        )

        pass_result = estimate_pass_probability(
            weighted,
            profile=prop,
            account=account,
            challenge=challenge,
            trials=mc_trials,
            fast=fast,
        )
        days_result = estimate_expected_pass_days(
            weighted,
            profile=prop,
            account=account,
            challenge=challenge,
            trials=max(100, mc_trials // 2),
            fast=fast,
        )

        metrics["pass_rate"] = pass_result.pass_probability
        metrics["fail_rate"] = pass_result.fail_probability
        metrics["avg_pass_days"] = days_result.expected_pass_days
        metrics["prob_recovery"] = max(0.0, 100.0 - pass_result.fail_probability)
        metrics["prob_ruin"] = pass_result.fail_probability
        return metrics
