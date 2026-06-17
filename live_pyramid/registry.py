"""Registry of active live pyramid sessions (bridge runtime singleton)."""

from __future__ import annotations

from live_pyramid.config import is_live_pyramid_enabled
from live_pyramid.evaluator import BarSnapshot, close_live_pyramid_session, evaluate_pyramid_on_bar, on_limit_filled
from live_pyramid.session import LivePyramidSession, WyckoffGateInput, create_live_pyramid_session


class LivePyramidRegistry:
    """trade_id / pyramid_group_id でセッションを管理。"""

    def __init__(self) -> None:
        self._by_trade: dict[str, LivePyramidSession] = {}
        self._by_group: dict[str, LivePyramidSession] = {}

    def __len__(self) -> int:
        return len(self._by_trade)

    def register(self, session: LivePyramidSession) -> LivePyramidSession:
        self._by_trade[session.trade_id] = session
        self._by_group[session.pyramid_group_id] = session
        return session

    def create_and_register(
        self,
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
    ) -> LivePyramidSession:
        if not is_live_pyramid_enabled(setup_type):
            raise RuntimeError(f"live pyramid disabled for setup_type={setup_type}")
        existing = self.get_by_trade_id(trade_id)
        if existing is not None and not existing.closed:
            return existing
        session = create_live_pyramid_session(
            trade_id=trade_id,
            setup_type=setup_type,
            symbol=symbol,
            direction=direction,
            entry=entry,
            sl=sl,
            tp=tp,
            atr=atr,
            initial_lot=initial_lot,
            base_ticket=base_ticket,
            entry_bar_index=entry_bar_index,
            daily_dd_remaining_percent=daily_dd_remaining_percent,
            trigger_r=trigger_r,
            kalman_velocity_at_entry=kalman_velocity_at_entry,
            ws_mode=ws_mode,
            pyramid_group_id=pyramid_group_id,
            tick_size=tick_size,
            tick_value=tick_value,
        )
        return self.register(session)

    def get_by_trade_id(self, trade_id: str) -> LivePyramidSession | None:
        return self._by_trade.get(trade_id)

    def get_by_group_id(self, pyramid_group_id: str) -> LivePyramidSession | None:
        return self._by_group.get(pyramid_group_id)

    def evaluate_tick(
        self,
        trade_id: str,
        bar: BarSnapshot,
        *,
        bar_index: int,
        daily_dd_remaining: float | None = None,
        ws_gates: WyckoffGateInput | None = None,
    ) -> list:
        session = self._require(trade_id)
        return evaluate_pyramid_on_bar(
            session,
            bar,
            bar_index=bar_index,
            daily_dd_remaining=daily_dd_remaining,
            ws_gates=ws_gates,
        )

    def notify_fill(
        self,
        trade_id: str,
        fill_price: float,
        *,
        position_ticket: int | None = None,
    ) -> list:
        session = self._require(trade_id)
        return on_limit_filled(session, fill_price, position_ticket=position_ticket)

    def close(self, trade_id: str) -> list:
        session = self._require(trade_id)
        actions = close_live_pyramid_session(session)
        self._by_trade.pop(trade_id, None)
        self._by_group.pop(session.pyramid_group_id, None)
        return actions

    def reset(self) -> None:
        self._by_trade.clear()
        self._by_group.clear()

    def _require(self, trade_id: str) -> LivePyramidSession:
        session = self.get_by_trade_id(trade_id)
        if session is None:
            raise KeyError(f"live pyramid session not found: {trade_id}")
        return session
