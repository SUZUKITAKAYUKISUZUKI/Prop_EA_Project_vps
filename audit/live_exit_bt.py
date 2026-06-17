"""Apply Live phase-2 exit rules (TP cap) in backtest L5 tracking."""

from __future__ import annotations

import os

from audit.live_tp_cap import is_live_tp_cap_enabled, resolve_live_take_profit

# Backward-compatible DBBS exit re-exports (canonical: strategies.dbbs_exit)
from strategies.dbbs_exit import (  # noqa: F401
    should_apply_dbbs_trailing_exit,
    track_dbbs_trailing_outcome,
)

should_use_dbbs_trailing = should_apply_dbbs_trailing_exit


def _env_flag(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("0", "false", "off", "no", "disabled"):
        return False
    if raw in ("1", "true", "yes", "on", "enabled"):
        return True
    return None


def bt_apply_live_exit_rules() -> bool:
    """When True, backtest L5 uses the same TP / DBBS trail rules as Live."""
    explicit = _env_flag("BT_APPLY_LIVE_EXIT")
    if explicit is not None:
        return explicit
    return is_live_tp_cap_enabled()


def resolve_bt_take_profit(
    entry: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    *,
    setup_type: str | None = None,
    setup: object | None = None,
) -> float:
    if not bt_apply_live_exit_rules():
        return take_profit
    tp, _, _, _ = resolve_live_take_profit(
        entry,
        stop_loss,
        take_profit,
        direction,
        setup_type=setup_type,
        setup=setup,
    )
    return tp
