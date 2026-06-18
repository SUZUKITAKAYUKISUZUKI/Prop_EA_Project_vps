"""Monte Carlo wrapper for PDTS scenario evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from audit.risk_manager import STARTING_EQUITY
from core.pass_probability import AccountSnapshot, ChallengeState, _simulate_forward
from prae.metrics import max_drawdown_r
from src.digital_twin._metrics_common import prepare_weighted_trades
from src.services.profile_service import ProfileContext

DEFAULT_TRIAL_COUNTS = (100, 500, 1000)


class MonteCarloWrapper:
    def run_paths(
        self,
        trades: pd.DataFrame,
        weights: dict[str, float],
        profile_ctx: ProfileContext,
        *,
        trial_counts: tuple[int, ...] = DEFAULT_TRIAL_COUNTS,
        seed: int = 42,
        account_state: str = "challenge",
        fast: bool = True,
    ) -> dict[str, Any]:
        weighted = prepare_weighted_trades(trades, weights)
        if weighted.empty:
            return {str(n): _empty_mc(n) for n in trial_counts}

        results: dict[str, Any] = {}
        for trials in trial_counts:
            if account_state == "challenge":
                results[str(trials)] = self._challenge_mc(
                    weighted, profile_ctx, trials=trials, seed=seed, fast=fast
                )
            else:
                results[str(trials)] = self._bootstrap_mc(weighted, trials=trials, seed=seed)
        return results

    def _challenge_mc(
        self,
        weighted: pd.DataFrame,
        profile_ctx: ProfileContext,
        *,
        trials: int,
        seed: int,
        fast: bool,
    ) -> dict[str, Any]:
        prop = profile_ctx.to_prop_profile()
        account = AccountSnapshot(
            equity=STARTING_EQUITY,
            balance=STARTING_EQUITY,
            peak_equity=STARTING_EQUITY,
            phase_start_equity=STARTING_EQUITY,
        )
        challenge = ChallengeState(0, 0.0, 0.0, 0.0)
        work = weighted.sort_values("timestamp").reset_index(drop=True)
        n = len(work)
        rng = np.random.default_rng(seed)
        pass_n = fail_n = 0
        pass_days: list[float] = []
        dd_vals: list[float] = []
        r_vals: list[float] = []

        for _ in range(trials):
            idx = rng.integers(0, n, size=n)
            sample = work.iloc[idx].reset_index(drop=True)
            res = _simulate_forward(
                sample,
                profile=prop,
                account=account,
                challenge=challenge,
                fast=fast,
            )
            if res["outcome"] == "pass":
                pass_n += 1
                if res["pass_days"] is not None:
                    pass_days.append(float(res["pass_days"]))
            elif res["outcome"] == "fail":
                fail_n += 1
            dd_vals.append(float(res.get("max_dd_pct") or 0.0))
            r_vals.append(float(sample["R"].sum()))

        dd_arr = np.asarray(dd_vals, dtype=np.float64)
        return {
            "trials": trials,
            "pass_rate": round(pass_n / trials * 100.0, 2),
            "fail_rate": round(fail_n / trials * 100.0, 2),
            "worst_dd": round(float(dd_arr.max()), 2) if len(dd_arr) else 0.0,
            "dd_95pct": round(float(np.percentile(dd_arr, 95)), 2) if len(dd_arr) else 0.0,
            "expected_r": round(float(np.mean(r_vals)), 2) if r_vals else 0.0,
            "expected_days": round(float(np.mean(pass_days)), 1) if pass_days else 0.0,
        }

    def _bootstrap_mc(
        self,
        weighted: pd.DataFrame,
        *,
        trials: int,
        seed: int,
    ) -> dict[str, Any]:
        work = weighted.sort_values("timestamp").reset_index(drop=True)
        n = len(work)
        rng = np.random.default_rng(seed)
        dd_vals: list[float] = []
        r_vals: list[float] = []
        ruin_n = 0

        for _ in range(trials):
            idx = rng.integers(0, n, size=n)
            sample = work.iloc[idx].reset_index(drop=True)
            r = sample["R"].astype(float)
            dd = max_drawdown_r(r)
            dd_vals.append(dd)
            r_vals.append(float(r.sum()))
            if dd >= 10.0:
                ruin_n += 1

        dd_arr = np.asarray(dd_vals, dtype=np.float64)
        return {
            "trials": trials,
            "pass_rate": round(max(0.0, 100.0 - ruin_n / trials * 100.0), 2),
            "fail_rate": round(ruin_n / trials * 100.0, 2),
            "worst_dd": round(float(dd_arr.max()), 2) if len(dd_arr) else 0.0,
            "dd_95pct": round(float(np.percentile(dd_arr, 95)), 2) if len(dd_arr) else 0.0,
            "expected_r": round(float(np.mean(r_vals)), 2) if r_vals else 0.0,
            "expected_days": 0.0,
        }


def _empty_mc(trials: int) -> dict[str, Any]:
    return {
        "trials": trials,
        "pass_rate": 0.0,
        "fail_rate": 0.0,
        "worst_dd": 0.0,
        "dd_95pct": 0.0,
        "expected_r": 0.0,
        "expected_days": 0.0,
    }
