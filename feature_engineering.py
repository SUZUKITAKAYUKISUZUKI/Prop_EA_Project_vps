"""
feature_engineering.py — backward-compatibility shim (v3.1)

Legacy imports continue to work. New code should use main_platform.py directly.

v3.5: PyramidManager 統合（PYRAMID_ENABLED=1 で L5 ピラミッド追跡）
v3.9: 本番 Strategy A (LSFC) のみ。`--strategy lsfc`。
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from audit import dd_throttling as audit_dd_throttle
from strategies.london_sweep_failure import (
    LsfcSetup,
    LondonSweepFailureStrategy,
    SETUP_TYPE as LSFC_SETUP_TYPE,
)

from main_platform import *  # noqa: F401,F403
from main_platform import (  # noqa: F401
    MAX_HOLDING_BARS,
    _apply_trade_outcome as _apply_trade_outcome_base,
    _resolve_track_start_index,
    merge_rule_base_l4_bypass_tags,
    resolve_model_version,
)

from pyramid_manager import (
    PyramidManager,
    PyramidPosition,
    is_pyramid_enabled,
    is_pyramid_enabled_for_pending,
    pyramid_result_to_record_fields,
    track_with_pyramid,
)

_PYRAMID_COLUMNS = [
    "pyramid_layers",
    "pyramid_entry_prices",
    "pyramid_lot_sizes",
    "final_sl_at_close",
    "peak_unrealized_r",
]
_FVG_HTF_COLUMNS = [
    "htf_counter_trend",
    "htf_lot_multiplier",
    "fvg_final_lot_factor",
]
for _col in _PYRAMID_COLUMNS + _FVG_HTF_COLUMNS:
    if _col not in CSV_COLUMNS:
        CSV_COLUMNS.append(_col)


def _apply_trade_outcome(
    pending: PendingEvaluation,
    account: AccountState,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    bar_minutes: int,
    *,
    max_holding_bars: int | None = None,
) -> dict[str, Any]:
    """
    Phase-2: L5 未来追跡（ピラミッド OFF 時は main_platform 委譲、ON 時は PyramidManager）。
    """
    if not is_pyramid_enabled_for_pending(pending):
        record = _apply_trade_outcome_base(
            pending,
            account,
            gbp_df,
            eur_df,
            bar_minutes,
            max_holding_bars=max_holding_bars,
        )
        record.update(
            {
                "pyramid_layers": 0,
                "pyramid_entry_prices": "[]",
                "pyramid_lot_sizes": "[]",
                "final_sl_at_close": round(pending.setup.stop_loss, 5),
                "peak_unrealized_r": 0.0,
            }
        )
        return record

    setup = pending.setup
    pair_df = gbp_df if uses_primary_dataframe(setup.pair) else eur_df
    take_profit = setup.take_profit
    if pending.setup_type == "CSPA":
        from strategies.archive.cspa import CspaSetup, scale_cspa_take_profit

        if isinstance(setup, CspaSetup):
            take_profit = scale_cspa_take_profit(
                setup.entry_price,
                setup.take_profit,
                setup.direction,
                pending.cspa_tp_multiplier,
            )
    risk = abs(setup.entry_price - setup.stop_loss)
    atr_price = risk * pending.atr_ratio if pending.atr_ratio > 0 else risk * 0.5
    sim_lot = pending.final_lot_size if pending.final_lot_size > 0 else (
        pending.lot_size if pending.lot_size > 0 else 1.0
    )

    holding_cap = max_holding_bars if max_holding_bars is not None else MAX_HOLDING_BARS
    pyramid_result = track_with_pyramid(
        pair_df,
        pending.start_idx,
        setup.direction,
        setup.entry_price,
        setup.stop_loss,
        take_profit,
        bar_minutes,
        sim_lot,
        atr_price,
        pending.daily_rem,
        trade_id=pending.trade_id,
        force_close_at_timeout=pending.force_close_at_timeout,
        timeout_server_hour=pending.timeout_server_hour,
        entry_timestamp=pd.Timestamp(setup.timestamp),
        max_holding_bars=holding_cap,
        setup_type=pending.setup_type,
        skip_strategy_enable_check=True,
    )

    shadow_result = pyramid_result.result if pyramid_result.result in ("WIN", "LOSS") else "LOSS"
    shadow_profit_r = pyramid_result.profit_r
    shadow_pips = pyramid_result.profit_pips
    holding = pyramid_result.holding_minutes

    executed = (not pending.is_reject) and pending.lot_factor > 0
    equity_before = pending.equity_before

    if executed:
        trade_result = shadow_result
        profit_r = shadow_profit_r
        profit_loss = shadow_pips
        equity_after = account.equity + pending.risk_budget * profit_r
        account.equity = equity_after
        shadow_result_out = "NONE"
        shadow_profit_r_out = 0.0

        if trade_result == "LOSS":
            audit_dd_throttle.register_executed_streak(account, won=False)
        else:
            audit_dd_throttle.register_executed_streak(account, won=True)

        if audit_rm.is_mutual_exclusion_enabled():
            account.register_executed_position(
                setup.timestamp,
                setup.pair,
                pending.setup_type,
                holding,
            )
    else:
        trade_result = "NOT_EXECUTED"
        profit_r = 0.0
        profit_loss = 0.0
        equity_after = equity_before
        shadow_result_out = shadow_result
        shadow_profit_r_out = shadow_profit_r
        if shadow_result == "WIN":
            account.consecutive_losses = 0

    record = {
        "trade_id": pending.trade_id,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "pair": setup.pair,
        "equity_before_trade": round(equity_before, 2),
        "equity_after_trade": round(equity_after, 2),
        "daily_dd_remaining_percent": round(pending.daily_rem, 4),
        "monthly_dd_remaining_percent": round(pending.monthly_rem, 4),
        "setup_type": pending.setup_type,
        "candidate_score": pending.candidate_score,
        "bayes_probability": pending.bayes_probability,
        "smt_intensity": round(pending.smt, 2),
        "model_version": resolve_model_version(pending.setup_type),
        "reason_codes": json.dumps(
            merge_rule_base_l4_bypass_tags(
                list(pending.tags),
                pending.setup_type,
                pending.decision_source,
                htf_trend_direction=pending.htf_trend_direction,
            ),
            ensure_ascii=False,
        ),
        "risk_score": pending.risk_score,
        "llm_latency_ms": pending.latency,
        "decision_source": pending.decision_source,
        "lot_factor": pending.lot_factor,
        "llm_score": pending.llm_confidence_score,
        "final_lot_size": round(pending.final_lot_size if pending.final_lot_size > 0 else pending.lot_size, 4),
        "entry_price": round(setup.entry_price, 5),
        "stop_loss": round(setup.stop_loss, 5),
        "take_profit": round(setup.take_profit, 5),
        "trade_result": trade_result,
        "profit_loss": round(profit_loss, 2),
        "profit_r": round(profit_r, 2),
        "holding_time": holding,
        "shadow_result": shadow_result_out,
        "shadow_profit_r": shadow_profit_r_out,
        "smt_diff": round(pending.smt_diff, 4),
        "smt_leader": pending.smt_leader,
        "wick_ratio_pct": round(getattr(setup, "wick_ratio_pct", 0.0), 4),
        "atr_ratio": round(pending.atr_ratio, 4),
        "has_bos": pending.has_bos,
        "vp_zone": pending.vp_zone,
        "l2_regime": pending.l2_regime,
        "l2_base_lot_factor": round(pending.l2_base_lot_factor, 4),
        "htf_trend": pending.htf_trend,
        "divergence_direction": pending.divergence_direction,
        "l4_multiplier": round(pending.l4_multiplier, 4),
        "l4_smt_interpretation": pending.l4_smt_interpretation,
        "htf_counter_trend": pending.htf_counter_trend,
        "htf_lot_multiplier": round(pending.htf_lot_multiplier, 4),
        "fvg_final_lot_factor": round(pending.fvg_final_lot_factor, 4),
        "ev_rank": round(
            pending.dn_ev_rank_v2
            if isinstance(setup, DiNapoliSetup) and pending.dn_prop_gate_tier
            else (
                pending.dn_ev_rank
                if isinstance(setup, DiNapoliSetup) and pending.dn_ev_bucket
                else pending.ttm_ev_rank
            ),
            6,
        ),
        "ev_lot_multiplier": round(
            pending.dn_prop_gate_lot_multiplier
            if isinstance(setup, DiNapoliSetup) and pending.dn_prop_gate_tier
            else pending.ttm_ev_lot_multiplier,
            4,
        ),
        "sized_result_r": round(profit_r, 4),
        **pyramid_result_to_record_fields(pyramid_result),
        "_setup": setup,
        "_pending": pending,
    }
    return record


__all__ = list(globals().keys())
