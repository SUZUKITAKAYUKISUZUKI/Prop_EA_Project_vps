"""
DN Prop Gate V1 — base_risk sweep with full Fintokei 2-phase simulator.

Pattern B tier sizing is fixed; only base_risk (decimal fraction) varies.
Uses Profit Cushion + Twin Brake identical to prop_audit_reporter / live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Match production defense stack used in prop_audit_reporter sliding sim.
os.environ.setdefault("PROFIT_CUSHION_ENABLED", "1")
os.environ.setdefault("TWIN_BRAKE_ENABLED", "1")

from audit.risk_manager import CHALLENGE_PROFIT_TARGET_PCT, STARTING_EQUITY
from prop_audit_reporter import (
    DAILY_DD_LIMIT_PCT,
    TOTAL_DD_LIMIT_PCT,
    WINDOW_DAYS,
    WINDOW_STEP_DAYS,
    _apply_dynamic_defense_multipliers,
    _simulated_daily_dd_remaining_percent,
    generate_window_starts,
)
from src.filters.dn_prop_gate_v1 import SIZING_PATTERNS, apply_sizing

PHASE2_TARGET_PCT = 6.0
PATTERN_B = SIZING_PATTERNS["B"]

BASE_RISK_CANDIDATES_PCT: tuple[float, ...] = (
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)


def base_risk_frac(pct: float) -> float:
    return float(pct) / 100.0


def prepare_pattern_b_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Map feature log rows to Fintokei sim columns (profit_r = Pattern B scaled R)."""
    sized = apply_sizing(df, PATTERN_B)
    out = sized.copy()
    out["profit_r"] = pd.to_numeric(out["scaled_r"], errors="coerce").fillna(0.0)
    out["lot_factor"] = 1.0
    return out


def _apply_trade_equity_fixed_base(
    equity: float,
    profit_r: float,
    lot_factor: float,
    base_risk: float,
) -> float:
    return equity * (1.0 + base_risk * lot_factor * profit_r)


def simulate_two_phase_fintokei(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    *,
    base_risk: float,
    max_calendar_days: int | None = WINDOW_DAYS,
) -> dict[str, Any]:
    if df.empty:
        return {
            "outcome": "timeout",
            "total_days": None,
            "max_dd": 0.0,
        }

    start = pd.Timestamp(window_start).normalize()
    equity = float(STARTING_EQUITY)
    phase_start = equity
    peak = equity
    phase = 1
    target_pct = CHALLENGE_PROFIT_TARGET_PCT
    target_equity = equity * (1.0 + target_pct / 100.0)
    max_dd = 0.0

    current_date: str | None = None
    daily_start = equity
    day_min = equity
    phase2_start_day: int | None = None

    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if ts.normalize() < start:
            continue
        if max_calendar_days is not None and (ts.normalize() - start).days > max_calendar_days:
            break

        date_key = ts.strftime("%Y-%m-%d")
        if date_key != current_date:
            current_date = date_key
            daily_start = equity
            day_min = equity

        daily_rem = _simulated_daily_dd_remaining_percent(daily_start, day_min)
        lot = _apply_dynamic_defense_multipliers(
            float(row.lot_factor),
            equity,
            phase_start,
            daily_rem,
            "challenge",
        )
        equity = _apply_trade_equity_fixed_base(
            equity,
            float(row.profit_r),
            lot,
            base_risk,
        )
        day_min = min(day_min, equity)
        peak = max(peak, equity)
        total_dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, total_dd)
        daily_dd = (daily_start - day_min) / daily_start * 100.0 if daily_start > 0 else 0.0

        if daily_dd >= DAILY_DD_LIMIT_PCT or total_dd >= TOTAL_DD_LIMIT_PCT:
            return {"outcome": "fail", "total_days": None, "max_dd": max_dd}

        if equity >= target_equity:
            day_count = max(0, (ts.normalize() - start).days)
            if phase == 1:
                phase = 2
                phase2_start_day = day_count
                equity = float(STARTING_EQUITY)
                phase_start = equity
                peak = equity
                target_pct = PHASE2_TARGET_PCT
                target_equity = equity * (1.0 + target_pct / 100.0)
                current_date = None
                continue
            return {"outcome": "pass", "total_days": float(day_count), "max_dd": max_dd}

    return {"outcome": "timeout", "total_days": None, "max_dd": max_dd}


@dataclass(frozen=True)
class RiskSweepSummary:
    base_risk_pct: float
    pass_rate: float
    avg_pass_days: float
    median_pass_days: float
    p25_pass_days: float
    p75_pass_days: float
    window_max_dd_pct: float
    pf: float
    total_r: float
    full_max_dd_pct: float
    calmar: float
    pass_count: int
    fail_count: int
    timeout_count: int
    total_windows: int
    efficiency_score: float


def _pf(series: pd.Series) -> float:
    r = pd.to_numeric(series, errors="coerce").fillna(0.0)
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def analyze_full_period_equity(df: pd.DataFrame, base_risk: float) -> dict[str, float]:
    if df.empty:
        return {"max_dd_pct": 0.0, "calmar": 0.0, "final_equity": STARTING_EQUITY}

    equity = STARTING_EQUITY
    peak = STARTING_EQUITY
    max_dd = 0.0
    current_date: str | None = None
    daily_start = equity
    day_min = equity
    phase_start = equity

    for row in df.itertuples(index=False):
        date_key = pd.Timestamp(row.timestamp).strftime("%Y-%m-%d")
        if date_key != current_date:
            current_date = date_key
            daily_start = equity
            day_min = equity

        daily_rem = _simulated_daily_dd_remaining_percent(daily_start, day_min)
        lot = _apply_dynamic_defense_multipliers(
            float(row.lot_factor),
            equity,
            phase_start,
            daily_rem,
            "challenge",
        )
        equity = _apply_trade_equity_fixed_base(
            equity,
            float(row.profit_r),
            lot,
            base_risk,
        )
        day_min = min(day_min, equity)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    span = max(1, (df["timestamp"].max() - df["timestamp"].min()).days)
    years = span / 365.25
    ann = ((equity / STARTING_EQUITY) ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else 0.0
    calmar = ann / max_dd if max_dd > 1e-9 else float("inf")
    return {
        "max_dd_pct": float(max_dd),
        "calmar": float(calmar) if np.isfinite(calmar) else 999.0,
        "final_equity": float(equity),
    }


def pass_efficiency_score(pass_rate: float, avg_pass_days: float, max_dd_pct: float) -> float:
    denom = max(avg_pass_days, 0.1) * 2.0 * max(max_dd_pct, 0.1) * 5.0
    return float(pass_rate * 100.0 / denom)


def run_base_risk_sweep(df: pd.DataFrame, base_risk_pct: float) -> RiskSweepSummary:
    sim_df = prepare_pattern_b_trades(df)
    base_risk = base_risk_frac(base_risk_pct)

    starts = generate_window_starts(sim_df["timestamp"], WINDOW_DAYS, WINDOW_STEP_DAYS)
    if not starts:
        starts = [sim_df["timestamp"].min().normalize()]

    pass_days: list[float] = []
    window_max_dds: list[float] = []
    pass_n = fail_n = timeout_n = 0

    for w_start in starts:
        w_end = w_start + pd.Timedelta(days=WINDOW_DAYS)
        w_df = sim_df[(sim_df["timestamp"] >= w_start) & (sim_df["timestamp"] < w_end)]
        res = simulate_two_phase_fintokei(w_df, w_start, base_risk=base_risk)
        window_max_dds.append(float(res["max_dd"]))
        if res["outcome"] == "pass":
            pass_n += 1
            if res["total_days"] is not None:
                pass_days.append(float(res["total_days"]))
        elif res["outcome"] == "fail":
            fail_n += 1
        else:
            timeout_n += 1

    arr = np.asarray(pass_days, dtype=np.float64)
    total = len(starts)
    pass_rate = (pass_n / total * 100.0) if total else 0.0
    avg_days = float(np.mean(arr)) if len(arr) else 0.0
    median_days = float(np.median(arr)) if len(arr) else 0.0
    p25 = float(np.percentile(arr, 25)) if len(arr) else 0.0
    p75 = float(np.percentile(arr, 75)) if len(arr) else 0.0
    window_max_dd = float(np.max(window_max_dds)) if window_max_dds else 0.0

    equity_stats = analyze_full_period_equity(sim_df, base_risk)
    pf = _pf(sim_df["profit_r"])
    total_r = float(sim_df["profit_r"].sum())
    score = pass_efficiency_score(pass_rate, avg_days, equity_stats["max_dd_pct"])

    return RiskSweepSummary(
        base_risk_pct=base_risk_pct,
        pass_rate=pass_rate,
        avg_pass_days=avg_days,
        median_pass_days=median_days,
        p25_pass_days=p25,
        p75_pass_days=p75,
        window_max_dd_pct=window_max_dd,
        pf=pf,
        total_r=total_r,
        full_max_dd_pct=equity_stats["max_dd_pct"],
        calmar=equity_stats["calmar"],
        pass_count=pass_n,
        fail_count=fail_n,
        timeout_count=timeout_n,
        total_windows=total,
        efficiency_score=score,
    )


def rank_wft_candidates(rows: list[RiskSweepSummary]) -> list[RiskSweepSummary]:
    return sorted(
        rows,
        key=lambda r: (
            -r.pass_rate,
            r.avg_pass_days,
            r.full_max_dd_pct,
            -r.pf,
            -r.total_r,
        ),
    )
