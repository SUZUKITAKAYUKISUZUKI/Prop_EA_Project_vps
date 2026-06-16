"""Phase 5.3 / 5.4 — Pass probability and expected pass days models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from audit.risk_manager import STARTING_EQUITY, effective_base_risk_pct
from core.prop_profiles import PropProfile
from core.progress_risk import apply_progress_risk_to_lot_factor
from prop_audit_reporter import _apply_dynamic_defense_multipliers, _simulated_daily_dd_remaining_percent


@dataclass(frozen=True)
class ChallengeState:
    days_elapsed: int
    profit_progress_percent: float
    daily_dd_used_percent: float
    total_dd_used_percent: float


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    balance: float
    peak_equity: float
    phase_start_equity: float = STARTING_EQUITY


@dataclass(frozen=True)
class PassProbabilityResult:
    pass_probability: float
    fail_probability: float
    timeout_probability: float
    trials: int


@dataclass(frozen=True)
class ExpectedPassDaysResult:
    expected_pass_days: float
    median_pass_days: float
    trades_per_day: float
    expectancy_r: float


def _simulate_forward(
    trades: pd.DataFrame,
    *,
    profile: PropProfile,
    account: AccountSnapshot,
    challenge: ChallengeState,
    global_risk_mult: float = 1.0,
    max_days: int | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    if trades.empty:
        return {"outcome": "timeout", "pass_days": None, "max_dd_pct": 0.0}

    phase_start = account.phase_start_equity
    equity = account.equity
    peak = max(account.peak_equity, equity)
    target_equity = phase_start * (1.0 + profile.target_profit / 100.0)

    window_max_dd = challenge.total_dd_used_percent
    current_date: str | None = None
    daily_start = equity
    day_min = equity
    start_ts = pd.Timestamp(trades["timestamp"].iloc[0])
    elapsed_days = challenge.days_elapsed

    for trade_idx, row in enumerate(trades.itertuples(index=False)):
        ts = pd.Timestamp(row.timestamp)
        date_key = ts.strftime("%Y-%m-%d")
        if date_key != current_date:
            current_date = date_key
            daily_start = equity
            day_min = equity

        daily_rem = _simulated_daily_dd_remaining_percent(daily_start, day_min)
        lot = float(getattr(row, "lot_factor", 1.0) or 1.0)
        lot *= global_risk_mult
        lot = apply_progress_risk_to_lot_factor(lot, challenge.profit_progress_percent)
        if fast:
            effective_lot = lot
        else:
            effective_lot = _apply_dynamic_defense_multipliers(
                lot,
                equity,
                phase_start,
                daily_rem,
                profile.profile_key,
                trades_in_phase=trade_idx,
            )
        base_risk = effective_base_risk_pct(profile.profile_key, phase_start, equity)
        profit_r = float(getattr(row, "R", getattr(row, "profit_r", 0.0)))
        equity *= 1.0 + base_risk * effective_lot * profit_r
        day_min = min(day_min, equity)
        peak = max(peak, equity)

        total_dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        window_max_dd = max(window_max_dd, total_dd)
        daily_dd = (
            (daily_start - day_min) / daily_start * 100.0 if daily_start > 0 else 0.0
        )

        if daily_dd >= profile.daily_dd_limit or total_dd >= profile.total_dd_limit:
            return {"outcome": "fail", "pass_days": None, "max_dd_pct": window_max_dd}

        progress = max(0.0, (equity - phase_start) / phase_start * 100.0)
        if equity >= target_equity or progress >= profile.target_profit:
            pass_days = elapsed_days + max(
                0, (ts.normalize() - start_ts.normalize()).days
            )
            if max_days and pass_days > max_days:
                return {"outcome": "timeout", "pass_days": None, "max_dd_pct": window_max_dd}
            return {"outcome": "pass", "pass_days": float(pass_days), "max_dd_pct": window_max_dd}

    return {"outcome": "timeout", "pass_days": None, "max_dd_pct": window_max_dd}


def estimate_pass_probability(
    trades: pd.DataFrame,
    *,
    profile: PropProfile,
    account: AccountSnapshot,
    challenge: ChallengeState,
    trials: int = 1000,
    seed: int = 42,
    global_risk_mult: float = 1.0,
    horizon_trades: int | None = None,
    fast: bool = False,
) -> PassProbabilityResult:
    """Monte Carlo P(pass before fail) from current challenge state."""
    if trades.empty or trials <= 0:
        return PassProbabilityResult(0.0, 0.0, 100.0, trials)

    work = trades.sort_values("timestamp").reset_index(drop=True)
    n = len(work)
    sample_len = min(n, horizon_trades or n)
    rng = np.random.default_rng(seed)
    pass_n = fail_n = timeout_n = 0

    for _ in range(trials):
        idx = rng.integers(0, n, size=sample_len)
        sample = work.iloc[idx].reset_index(drop=True)
        res = _simulate_forward(
            sample,
            profile=profile,
            account=account,
            challenge=challenge,
            global_risk_mult=global_risk_mult,
            max_days=profile.max_days or None,
            fast=fast,
        )
        if res["outcome"] == "pass":
            pass_n += 1
        elif res["outcome"] == "fail":
            fail_n += 1
        else:
            timeout_n += 1

    return PassProbabilityResult(
        pass_probability=round(pass_n / trials * 100.0, 2),
        fail_probability=round(fail_n / trials * 100.0, 2),
        timeout_probability=round(timeout_n / trials * 100.0, 2),
        trials=trials,
    )


def estimate_expected_pass_days(
    trades: pd.DataFrame,
    *,
    profile: PropProfile,
    account: AccountSnapshot,
    challenge: ChallengeState,
    trials: int = 500,
    seed: int = 42,
    global_risk_mult: float = 1.0,
    horizon_trades: int | None = None,
    fast: bool = False,
) -> ExpectedPassDaysResult:
    """Estimate expected pass days using trade frequency + forward simulation."""
    if trades.empty:
        return ExpectedPassDaysResult(0.0, 0.0, 0.0, 0.0)

    work = trades.sort_values("timestamp").reset_index(drop=True)
    span_days = max(
        1,
        (work["timestamp"].max() - work["timestamp"].min()).days or 1,
    )
    trades_per_day = len(work) / span_days
    expectancy_r = float(work["R"].mean()) if "R" in work.columns else float(work.get("profit_r", pd.Series([0])).mean())

    rng = np.random.default_rng(seed)
    n = len(work)
    sample_len = min(n, horizon_trades or min(n, 200))
    pass_days: list[float] = []

    for _ in range(trials):
        idx = rng.integers(0, n, size=sample_len)
        sample = work.iloc[idx].reset_index(drop=True)
        res = _simulate_forward(
            sample,
            profile=profile,
            account=account,
            challenge=challenge,
            global_risk_mult=global_risk_mult,
            max_days=profile.max_days or None,
            fast=fast,
        )
        if res["outcome"] == "pass" and res["pass_days"] is not None:
            pass_days.append(float(res["pass_days"]))

    arr = np.asarray(pass_days, dtype=np.float64)
    if len(arr) == 0:
        remaining = max(0.0, profile.target_profit - challenge.profit_progress_percent)
        proxy = remaining / max(expectancy_r * trades_per_day * 0.25, 0.01)
        return ExpectedPassDaysResult(
            expected_pass_days=round(proxy, 1),
            median_pass_days=round(proxy, 1),
            trades_per_day=round(trades_per_day, 3),
            expectancy_r=round(expectancy_r, 4),
        )

    return ExpectedPassDaysResult(
        expected_pass_days=round(float(np.mean(arr)), 1),
        median_pass_days=round(float(np.median(arr)), 1),
        trades_per_day=round(trades_per_day, 3),
        expectancy_r=round(expectancy_r, 4),
    )
