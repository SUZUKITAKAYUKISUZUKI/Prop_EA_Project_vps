"""DBBS exit logic — H1 BB20 trail (SL-first, -1R floor), Live signal + BT L5."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from strategies.dbbs import DbbsSetup, SETUP_TYPE as DBBS_SETUP_TYPE, evaluate_squeeze_trailing_exit
from strategies.dbbs_common import (
    BB_PERIOD_SHORT,
    BB_STD_MULT,
    DBBS_MAX_LOSS_R,
    SQUEEZE_MAX_HOLD_H1,
    SQUEEZE_MIN_HOLD_H1,
    precompute_bb_series,
)
from strategies.market_utils import pip_size_for_pair

DBBS_TRAIL_ENV = "DBBS_LIVE_TRAIL_ENABLED"
DBBS_TRAIL_DEFAULT = "1"


def is_dbbs_live_trail_enabled() -> bool:
    raw = os.getenv(DBBS_TRAIL_ENV, DBBS_TRAIL_DEFAULT).strip().lower()
    return raw in ("1", "true", "yes", "on", "enabled")


def _backtest_mode_enabled() -> bool:
    return os.getenv("BACKTEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def should_apply_dbbs_trailing_exit(
    setup_type: str | None,
    setup: object | None,
    *,
    for_backtest: bool | None = None,
) -> bool:
    """True when DBBS H1 trailing exit replaces fixed SL/TP tracking."""
    if not is_dbbs_live_trail_enabled():
        return False
    if setup_type != DBBS_SETUP_TYPE or not isinstance(setup, DbbsSetup):
        return False
    if for_backtest is None:
        for_backtest = _backtest_mode_enabled()
    if for_backtest:
        from audit.live_exit_bt import bt_apply_live_exit_rules

        return bt_apply_live_exit_rules()
    return True


def track_dbbs_trailing_outcome(
    setup: DbbsSetup,
    h1_df: pd.DataFrame,
) -> tuple[str, float, int]:
    """Simulate DBBS H1 trailing exit. Returns (trade_result, profit_r, holding_minutes)."""
    from strategies.bt_ohlcv import as_ohlcv

    arr = as_ohlcv(h1_df)
    close = arr.close
    if len(close) < BB_PERIOD_SHORT + 5:
        return "LOSS", -DBBS_MAX_LOSS_R, 0

    upper, middle, lower, _width = precompute_bb_series(close, BB_PERIOD_SHORT, BB_STD_MULT)
    std = (upper - middle) / 2.0
    start = int(setup.bar_index_h1)
    if start < 0 or start >= len(close):
        start = max(0, min(len(close) - 1, start))

    label, result_r, held_h1 = evaluate_squeeze_trailing_exit(
        setup=setup,
        h1_close=close,
        bb20_middle=middle,
        bb20_std=std,
        start_h1_index=start,
    )
    holding_minutes = int(held_h1) * 60
    return label, float(result_r), holding_minutes


def resolve_dbbs_l5_outcome(
    setup: DbbsSetup,
    h1_df: pd.DataFrame,
) -> tuple[str, float, float, int]:
    """BT/L5 unified DBBS exit: result, profit_r, shadow_pips, holding_minutes."""
    shadow_result, shadow_profit_r, holding = track_dbbs_trailing_outcome(setup, h1_df)
    risk = abs(float(setup.entry_price) - float(setup.stop_loss))
    pip = pip_size_for_pair(setup.pair)
    shadow_pips = shadow_profit_r * risk / pip if risk > 0 and pip > 0 else 0.0
    return shadow_result, shadow_profit_r, float(shadow_pips), holding


def build_dbbs_exit_signal_fields(setup: DbbsSetup | None = None) -> dict[str, Any]:
    """Flat JSON keys for MT5 DbbsExitManager registration."""
    if not is_dbbs_live_trail_enabled():
        return {"setup_type": DBBS_SETUP_TYPE, "exit_mode": "FIXED_SL_TP"}

    fields: dict[str, Any] = {
        "setup_type": DBBS_SETUP_TYPE,
        "exit_mode": "DBBS_TRAIL",
        "exit_min_hold_h1": int(SQUEEZE_MIN_HOLD_H1),
        "exit_max_hold_h1": int(SQUEEZE_MAX_HOLD_H1),
        "exit_trail_sigma": 1.0,
        "exit_max_loss_r": DBBS_MAX_LOSS_R,
    }
    if setup is not None:
        fields["entry_price"] = round(float(setup.entry_price), 5)
        fields["stop_loss"] = round(float(setup.stop_loss), 5)
    return fields
