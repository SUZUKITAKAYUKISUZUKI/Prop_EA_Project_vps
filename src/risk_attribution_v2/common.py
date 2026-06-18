"""Shared utilities for Portfolio Risk Attribution Engine v2."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prae.loaders import normalize_trade_frame
from prae.metrics import max_drawdown_r, profit_factor, recovery_factor, sharpe_r
from src.database.profile_migrations import SETUP_TYPE_BY_STRATEGY_CODE

SETUP_TO_CODE: dict[str, str] = {v: k for k, v in SETUP_TYPE_BY_STRATEGY_CODE.items()}

RISK_SCORE_WEIGHTS = {
    "dd": 0.40,
    "ulcer": 0.35,
    "losing_streak": 0.25,
}


def prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    work = normalize_trade_frame(trades) if "R" not in trades.columns else trades.copy()
    work["strategy_code"] = work.get("setup_type", work["strategy"]).map(SETUP_TO_CODE).fillna(
        work.get("strategy", work.get("setup_type"))
    )
    if "setup_type" not in work.columns:
        work["setup_type"] = work["strategy"]
    work["pair"] = work.get("pair", work.get("symbol", "UNKNOWN")).astype(str).str.upper()
    if "timestamp" not in work.columns:
        raise ValueError("Trade frame requires timestamp")
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if "session" not in work.columns:
        work["session"] = work["timestamp"].dt.hour.map(_session_from_hour)
    if "direction" not in work.columns:
        work["direction"] = np.where(work["R"] >= 0, "BUY", "SELL")
    return work


def _session_from_hour(hour: int) -> str:
    if 0 <= hour < 8:
        return "ASIA"
    if 8 <= hour < 16:
        return "LONDON"
    return "NY"


def win_rate_pct(r: pd.Series) -> float:
    if r.empty:
        return 0.0
    return round(float((r > 0).sum() / len(r) * 100.0), 2)


def ulcer_index(r: pd.Series) -> float:
    arr = r.astype(float).to_numpy()
    if arr.size == 0:
        return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return round(float(np.sqrt(np.mean(np.square(dd)))), 4)


def max_losing_streak(r: pd.Series) -> int:
    streak = 0
    best = 0
    for val in r.astype(float):
        if val < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def strategy_metrics(r: pd.Series) -> dict[str, Any]:
    r = r.astype(float)
    total_r = round(float(r.sum()), 2)
    dd = round(max_drawdown_r(r), 2)
    pf_val = profit_factor(r)
    pf = round(pf_val, 4) if pf_val != float("inf") else 999.0
    rf_val = recovery_factor(total_r, dd)
    rf = round(rf_val, 4) if rf_val != float("inf") else 999.0
    return {
        "total_r": total_r,
        "pf": pf,
        "max_dd": dd,
        "ulcer": ulcer_index(r),
        "sharpe": round(sharpe_r(r), 4),
        "recovery_factor": rf,
        "win_rate": win_rate_pct(r),
        "trade_count": int(len(r)),
        "max_losing_streak": max_losing_streak(r),
    }


def compute_risk_score(
    *,
    max_dd: float,
    ulcer: float,
    max_losing_streak: int,
    dd_scale: float = 1.0,
    ulcer_scale: float = 1.0,
    streak_scale: float = 1.0,
) -> float:
    dd_norm = min(100.0, max_dd / max(dd_scale, 1e-9) * 100.0)
    ulcer_norm = min(100.0, ulcer / max(ulcer_scale, 1e-9) * 100.0)
    streak_norm = min(100.0, max_losing_streak / max(streak_scale, 1e-9) * 100.0)
    score = (
        RISK_SCORE_WEIGHTS["dd"] * dd_norm
        + RISK_SCORE_WEIGHTS["ulcer"] * ulcer_norm
        + RISK_SCORE_WEIGHTS["losing_streak"] * streak_norm
    )
    return round(score, 1)


def normalize_contribution_pct(values: dict[str, float]) -> dict[str, float]:
    total = sum(abs(v) for v in values.values())
    if total <= 0:
        n = len(values) or 1
        return {k: round(100.0 / n, 1) for k in values}
    out = {k: round(abs(v) / total * 100.0, 1) for k, v in values.items()}
    drift = round(100.0 - sum(out.values()), 1)
    if out and abs(drift) >= 0.1:
        first = next(iter(out))
        out[first] = round(out[first] + drift, 1)
    return out


def loss_contribution(rows: pd.DataFrame, dimension: str) -> dict[str, float]:
    losses = rows.loc[rows["R"] < 0]
    if losses.empty:
        return {}
    grouped = losses.groupby(dimension)["R"].sum().abs()
    return normalize_contribution_pct(grouped.to_dict())
