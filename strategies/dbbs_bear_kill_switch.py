"""
DBBS Bear Kill Switch V2 — default production edge-state risk control.

    if (last_3_avg_r <= 0.20) risk = 0.00;
    else risk = 1.00;

`last_3_avg_r` uses the mean ``result_r`` of the prior 3 closed DBBS trades (strictly pre-trade).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

BEAR_KILL_SWITCH_V2_THRESHOLD = 0.20
BEAR_KILL_SWITCH_V2_LOOKBACK = 3
DECISION_SOURCE_ACTIVE = "BEAR_KILL_SWITCH_V2"
DECISION_SOURCE_ALLOW = "ALLOW"

_tracker: "DbbsEdgeStateTracker | None" = None


def bear_kill_switch_threshold() -> float:
    raw = os.getenv("DBBS_BEAR_KILL_SWITCH_THRESHOLD", str(BEAR_KILL_SWITCH_V2_THRESHOLD))
    try:
        return float(raw)
    except ValueError:
        return BEAR_KILL_SWITCH_V2_THRESHOLD


def is_bear_kill_switch_enabled() -> bool:
    raw = os.getenv("DBBS_BEAR_KILL_SWITCH", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class DbbsEdgeStateTracker:
    """Portfolio-global realized-R memory for DBBS (chronological across pairs)."""

    threshold: float = BEAR_KILL_SWITCH_V2_THRESHOLD
    lookback: int = BEAR_KILL_SWITCH_V2_LOOKBACK
    _history: list[float] | None = None

    def __post_init__(self) -> None:
        if self._history is None:
            self._history = []

    def reset(self) -> None:
        self._history.clear()

    def last_3_avg_r(self) -> float:
        if not self._history:
            return float("nan")
        window = self._history[-self.lookback :]
        return float(np.mean(window))

    def risk_multiplier(self) -> float:
        if not is_bear_kill_switch_enabled():
            return 1.0
        last3 = self.last_3_avg_r()
        if not np.isfinite(last3):
            return 1.0
        return 0.0 if last3 <= self.threshold else 1.0

    def is_kill_active(self) -> bool:
        return is_bear_kill_switch_enabled() and self.risk_multiplier() == 0.0

    def pre_trade_snapshot(self) -> dict[str, Any]:
        last3 = self.last_3_avg_r()
        mult = self.risk_multiplier()
        return {
            "last_3_avg_r": last3,
            "edge_risk_mult": mult,
            "bear_kill_switch_active": bool(mult == 0.0),
        }

    def record_result(self, result_r: float) -> None:
        self._history.append(float(result_r))

    def apply_to_trade(self, raw_result_r: float) -> tuple[float, str, dict[str, Any]]:
        snap = self.pre_trade_snapshot()
        mult = float(snap["edge_risk_mult"])
        sized = float(raw_result_r) * mult
        source = DECISION_SOURCE_ACTIVE if snap["bear_kill_switch_active"] else DECISION_SOURCE_ALLOW
        self.record_result(raw_result_r)
        return sized, source, snap


def get_edge_tracker(*, reset: bool = False) -> DbbsEdgeStateTracker:
    global _tracker
    if reset or _tracker is None:
        _tracker = DbbsEdgeStateTracker(threshold=bear_kill_switch_threshold())
    return _tracker


def reset_edge_tracker() -> DbbsEdgeStateTracker:
    return get_edge_tracker(reset=True)


def record_closed_trade_result(result_r: float) -> None:
    """Call when a DBBS trade closes (live EA / bridge) to update edge memory."""
    get_edge_tracker().record_result(float(result_r))
