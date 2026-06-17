"""Backward-compatible re-exports — Fintokei rules live in audit.risk_manager."""

from __future__ import annotations

from audit.risk_manager import (
    FINTOKEI_SINGLE_POSITION_LOSS_LIMIT_PCT,
    REASON_FINTOKEI_SINGLE_POSITION_CAP,
    cap_lot_factor_to_fintokei_single_position,
    evaluate_fintokei_trade_fail_reason,
    exceeds_fintokei_single_position_limit,
    fintokei_single_position_loss_limit_pct,
    is_fintokei_single_position_rule_enabled,
    single_trade_loss_pct,
)

# Legacy alias
cap_lot_factor_to_single_position_limit = cap_lot_factor_to_fintokei_single_position
exceeds_single_position_loss_limit = exceeds_fintokei_single_position_limit

__all__ = [
    "FINTOKEI_SINGLE_POSITION_LOSS_LIMIT_PCT",
    "REASON_FINTOKEI_SINGLE_POSITION_CAP",
    "cap_lot_factor_to_fintokei_single_position",
    "cap_lot_factor_to_single_position_limit",
    "evaluate_fintokei_trade_fail_reason",
    "exceeds_fintokei_single_position_limit",
    "exceeds_single_position_loss_limit",
    "fintokei_single_position_loss_limit_pct",
    "is_fintokei_single_position_rule_enabled",
    "single_trade_loss_pct",
]
