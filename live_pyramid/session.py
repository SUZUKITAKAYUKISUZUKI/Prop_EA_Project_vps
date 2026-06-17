"""In-memory live pyramid session (PyramidManager mirror + pending Limit state)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from pyramid_manager import PyramidManager

from live_pyramid.config import (
    LIVE_PYRAMID_TRIGGER_R,
    resolve_live_max_pyramid_layers,
    resolve_live_pyramid_trigger_r,
)
from live_pyramid.limit_order import PyramidLimitIntent


@dataclass
class PendingLimitState:
    limit_price: float
    lot_size: float
    unified_sl: float
    layer_index: int
    placed_at_bar_index: int
    ttl_bars: int
    order_ticket: int | None = None


@dataclass
class WyckoffGateInput:
    """WR ピラミッド補助入力（タイムリミットのみ — カルマンフィルタは廃止）。"""

    past_time_limit: bool = False


@dataclass
class LivePyramidSession:
    trade_id: str
    pyramid_group_id: str
    setup_type: str
    symbol: str
    direction: str
    mgr: PyramidManager
    entry_bar_index: int = 0
    current_bar_index: int = 0
    base_ticket: int | None = None
    position_tickets: list[int] = field(default_factory=list)
    pending_limit: PendingLimitState | None = None
    last_rejected_reason: str = ""
    time_limit_triggered: bool = False
    decel_exit_triggered: bool = False
    kalman_velocity_at_entry: float = 0.0
    trigger_r: float = LIVE_PYRAMID_TRIGGER_R
    ws_mode: bool = False
    closed: bool = False
    equity: float = 0.0
    phase_start_equity: float = 0.0
    stats_limit_placed: int = 0
    stats_limit_filled: int = 0
    stats_limit_cancelled: int = 0
    stats_market_fallback: int = 0

    @property
    def bars_since_entry(self) -> int:
        return max(0, self.current_bar_index - self.entry_bar_index)

    @property
    def upper_layer_tickets(self) -> list[int]:
        if len(self.position_tickets) <= 1:
            return []
        return list(self.position_tickets[1:])


def create_live_pyramid_session(
    *,
    trade_id: str,
    setup_type: str,
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    atr: float,
    initial_lot: float,
    base_ticket: int,
    entry_bar_index: int = 0,
    daily_dd_remaining_percent: float = 5.0,
    trigger_r: float | None = None,
    kalman_velocity_at_entry: float = 0.0,
    ws_mode: bool = False,
    pyramid_group_id: str | None = None,
    tick_size: float = 0.0,
    tick_value: float = 0.0,
    equity: float = 0.0,
    phase_start_equity: float = 0.0,
) -> LivePyramidSession:
    base_risk = abs(entry - sl)
    safe_atr = max(atr, base_risk * 0.01) if base_risk > 0 else atr
    mgr = PyramidManager(
        trade_id=trade_id,
        direction=direction,
        atr=safe_atr,
        base_risk=base_risk,
        breakeven_price=entry,
        take_profit=tp,
        initial_lot=initial_lot,
        initial_stop_loss=sl,
        max_pyramid_layers=resolve_live_max_pyramid_layers(setup_type),
        daily_dd_remaining_percent=daily_dd_remaining_percent,
        symbol=symbol,
        tick_size=tick_size,
        tick_value=tick_value,
    )
    group_id = pyramid_group_id or uuid.uuid4().hex[:12]
    return LivePyramidSession(
        trade_id=trade_id,
        pyramid_group_id=group_id,
        setup_type=setup_type,
        symbol=symbol,
        direction=direction.upper(),
        mgr=mgr,
        entry_bar_index=entry_bar_index,
        current_bar_index=entry_bar_index,
        base_ticket=base_ticket,
        position_tickets=[base_ticket],
        trigger_r=trigger_r if trigger_r is not None else resolve_live_pyramid_trigger_r(setup_type),
        kalman_velocity_at_entry=kalman_velocity_at_entry,
        ws_mode=ws_mode,
        equity=equity,
        phase_start_equity=phase_start_equity,
    )


def attach_pending_limit(
    session: LivePyramidSession,
    intent: PyramidLimitIntent,
    *,
    order_ticket: int | None = None,
) -> PendingLimitState:
    pending = PendingLimitState(
        limit_price=intent.limit_price,
        lot_size=intent.lot_size,
        unified_sl=intent.unified_sl,
        layer_index=intent.layer_index,
        placed_at_bar_index=session.current_bar_index,
        ttl_bars=intent.ttl_bars,
        order_ticket=order_ticket,
    )
    session.pending_limit = pending
    return pending
