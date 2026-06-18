"""Shared scenario metrics helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prae.metrics import apply_allocation_weights, max_drawdown_r, profit_factor, recovery_factor, sharpe_r, summarize_r
from src.risk_attribution_v2.common import compute_risk_score, strategy_metrics, ulcer_index as r_ulcer_index


def prepare_weighted_trades(trades: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    work = trades.copy()
    if "strategy" not in work.columns:
        work["strategy"] = work.get("strategy_code", work.get("setup_type"))
    weighted = apply_allocation_weights(work, weights)
    return weighted[weighted["allocation_weight"] > 0].copy()


def compute_scenario_metrics(
    trades: pd.DataFrame,
    weights: dict[str, float],
    *,
    health_score: float = 50.0,
) -> dict[str, Any]:
    active = prepare_weighted_trades(trades, weights)
    if active.empty:
        return _empty_metrics(health_score)

    r = active["R"].astype(float)
    lots = active.get("lot_factor", pd.Series(1.0, index=active.index))
    summary = summarize_r(r, active.get("timestamp"), lots, fast=True)
    dd_r = max_drawdown_r(r)
    pf_val = profit_factor(r)
    pf = pf_val if np.isfinite(pf_val) else 3.0
    rec = recovery_factor(float(summary.total_r), dd_r)
    rec_out = round(rec, 4) if np.isfinite(rec) else 0.0

    equity = np.cumsum(r.values)
    ulcer = r_ulcer_index(r)
    strat_metrics = strategy_metrics(r)
    risk = compute_risk_score(
        max_dd=strat_metrics["max_dd"],
        ulcer=ulcer,
        max_losing_streak=strat_metrics["max_losing_streak"],
        dd_scale=max(strat_metrics["max_dd"], 1.0),
        ulcer_scale=max(ulcer, 1.0),
        streak_scale=max(strat_metrics["max_losing_streak"], 1.0),
    )

    wins = int((r > 0).sum())
    win_rate = round(wins / len(r) * 100.0, 2) if len(r) else 0.0

    return {
        "pf": round(pf, 4),
        "total_r": round(float(summary.total_r), 2),
        "win_rate": win_rate,
        "sharpe": round(float(summary.sharpe), 4),
        "recovery_factor": rec_out,
        "ulcer_index": round(float(ulcer), 4),
        "risk_score": round(float(risk), 2),
        "max_dd": round(float(summary.max_dd_r), 2),
        "max_dd_pct": round(float(summary.max_dd_pct), 2),
        "health_score": round(float(health_score), 1),
        "trades": int(len(r)),
    }


def _empty_metrics(health_score: float) -> dict[str, Any]:
    return {
        "pf": 0.0,
        "total_r": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "recovery_factor": 0.0,
        "ulcer_index": 0.0,
        "risk_score": 0.0,
        "max_dd": 0.0,
        "max_dd_pct": 0.0,
        "health_score": round(float(health_score), 1),
        "trades": 0,
    }
