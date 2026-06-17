"""
Live forward TP cap — shrink distant take-profit targets sent to MT5.

Backtest paths are unchanged; only ``pending_to_trade_signal`` applies the cap.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

DEFAULT_LIVE_TP_MAX_R = 1.5
DINAPOLI_SETUP_TYPE = "DINAPOLI_STRUCTURE"
VAMR_SETUP_TYPE = "VAMR"
MIN_STRUCTURAL_TP_R = 0.5


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("0", "false", "off", "no", "disabled"):
        return False
    if raw in ("1", "true", "yes", "on", "enabled"):
        return True
    return default


def is_live_tp_cap_enabled() -> bool:
    return _env_flag("LIVE_TP_CAP_ENABLED", True)


def live_tp_max_r(setup_type: str | None = None) -> float:
    """Resolve max TP distance in R (setup-specific env overrides global)."""
    if setup_type:
        key = f"LIVE_TP_MAX_R_{setup_type.upper()}"
        raw = os.getenv(key, "").strip()
        if raw:
            try:
                return max(0.1, float(raw))
            except ValueError:
                pass
    raw = os.getenv("LIVE_TP_MAX_R", str(DEFAULT_LIVE_TP_MAX_R)).strip()
    try:
        return max(0.1, float(raw))
    except ValueError:
        return DEFAULT_LIVE_TP_MAX_R


def cap_live_take_profit(
    entry: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    *,
    setup_type: str | None = None,
    max_r: float | None = None,
) -> tuple[float, float, bool]:
    """
    Cap take-profit distance to ``max_r`` × risk.

    Returns (tp, tp_r, was_capped).
    """
    side = str(direction).upper()
    risk = abs(entry - stop_loss)
    if risk <= 0.0 or take_profit <= 0.0:
        return take_profit, 0.0, False

    limit = max_r if max_r is not None else live_tp_max_r(setup_type)

    if side == "BUY":
        reward = take_profit - entry
        if reward <= 0.0:
            return take_profit, reward / risk, False
        tp_r = reward / risk
        if tp_r <= limit:
            return take_profit, tp_r, False
        return entry + limit * risk, limit, True

    if side == "SELL":
        reward = entry - take_profit
        if reward <= 0.0:
            return take_profit, reward / risk, False
        tp_r = reward / risk
        if tp_r <= limit:
            return take_profit, tp_r, False
        return entry - limit * risk, limit, True

    return take_profit, 0.0, False


def _reward_r(entry: float, target: float, direction: str, risk: float) -> float:
    if risk <= 0.0:
        return 0.0
    side = str(direction).upper()
    if side == "BUY":
        return (target - entry) / risk
    if side == "SELL":
        return (entry - target) / risk
    return 0.0


def _pick_structural_target(
    entry: float,
    stop_loss: float,
    direction: str,
    candidates: list[float],
    *,
    max_r: float,
) -> float | None:
    """Return nearest valid structural target within [min, max_r]."""
    risk = abs(entry - stop_loss)
    if risk <= 0.0:
        return None
    best: float | None = None
    best_r = float("inf")
    for price in candidates:
        if price <= 0.0 or not np.isfinite(price):
            continue
        r = _reward_r(entry, price, direction, risk)
        if r < MIN_STRUCTURAL_TP_R or r > max_r:
            continue
        if r < best_r:
            best = price
            best_r = r
    return best


def resolve_structural_live_tp(
    entry: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    *,
    setup_type: str | None = None,
    setup: Any | None = None,
    max_r: float | None = None,
) -> tuple[float, str]:
    """VAMR / DiNapoli: prefer reachable structural targets before hard cap."""
    limit = max_r if max_r is not None else live_tp_max_r(setup_type)
    risk = abs(entry - stop_loss)
    if risk <= 0.0:
        return take_profit, "unchanged"

    st = str(setup_type or "")
    if st == DINAPOLI_SETUP_TYPE and setup is not None:
        cop = float(getattr(setup, "cop", 0.0) or 0.0)
        op = float(getattr(setup, "op", take_profit) or take_profit)
        picked = _pick_structural_target(entry, stop_loss, direction, [cop, op, take_profit], max_r=limit)
        if picked is not None:
            tag = "cop" if abs(picked - cop) < 1e-9 else "op"
            return picked, f"dn_{tag}"

    if st == VAMR_SETUP_TYPE and setup is not None:
        poc = float(getattr(setup, "poc", take_profit) or take_profit)
        vah = float(getattr(setup, "vah", 0.0) or 0.0)
        val = float(getattr(setup, "val", 0.0) or 0.0)
        side = str(direction).upper()
        order = [poc, vah] if side == "BUY" else [poc, val]
        picked = _pick_structural_target(entry, stop_loss, direction, order, max_r=limit)
        if picked is not None:
            tag = "poc" if abs(picked - poc) < 1e-9 else "va"
            return picked, f"vamr_{tag}"

    return take_profit, "default"


def resolve_live_take_profit(
    entry: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    *,
    setup_type: str | None = None,
    setup: Any | None = None,
) -> tuple[float, float, bool, str]:
    """Structural pick → cap. Returns (tp, tp_r, was_capped, tag)."""
    if not is_live_tp_cap_enabled():
        risk = abs(entry - stop_loss)
        return take_profit, _reward_r(entry, take_profit, direction, risk), False, "off"

    tp_work, tag = resolve_structural_live_tp(
        entry,
        stop_loss,
        take_profit,
        direction,
        setup_type=setup_type,
        setup=setup,
    )
    tp_out, tp_r, capped = cap_live_take_profit(
        entry,
        stop_loss,
        tp_work,
        direction,
        setup_type=setup_type,
    )
    if capped and tag == "default":
        tag = "capped"
    elif capped:
        tag = f"{tag}_capped"
    return tp_out, tp_r, capped, tag
