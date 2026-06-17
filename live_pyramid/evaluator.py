"""Bar-close pyramid evaluation → BridgeAction list (Limit conversion layer)."""

from __future__ import annotations

from dataclasses import dataclass

from live_pyramid.actions import BridgeAction
from live_pyramid.config import LIVE_PYRAMID_LIMIT_TTL_BARS, LIVE_PYRAMID_USE_MARKET_FALLBACK
from live_pyramid.limit_order import PyramidLimitIntent, build_pyramid_limit_intent
from live_pyramid.session import LivePyramidSession, PendingLimitState, WyckoffGateInput, attach_pending_limit
from live_pyramid.sizing import cap_pyramid_lot_size


@dataclass(frozen=True)
class BarSnapshot:
    open: float
    high: float
    low: float
    close: float


def _session_meta(session: LivePyramidSession) -> dict[str, str | float | bool | None]:
    return {
        "ws_kalman_velocity": session.kalman_velocity_at_entry,
        "ws_decel_exit": session.decel_exit_triggered,
        "ws_time_limit_exit": session.time_limit_triggered,
        "ws_pyramid_rejected_reason": session.last_rejected_reason or None,
    }


def _base_action(session: LivePyramidSession, action: str, message: str = "") -> BridgeAction:
    meta = _session_meta(session)
    return BridgeAction(
        action=action,  # type: ignore[arg-type]
        trade_id=session.trade_id,
        pyramid_group_id=session.pyramid_group_id,
        setup_type=session.setup_type,
        symbol=session.symbol,
        direction=session.direction,
        message=message,
        ws_kalman_velocity=meta["ws_kalman_velocity"],  # type: ignore[arg-type]
        ws_decel_exit=meta["ws_decel_exit"],  # type: ignore[arg-type]
        ws_time_limit_exit=meta["ws_time_limit_exit"],  # type: ignore[arg-type]
        ws_pyramid_rejected_reason=meta["ws_pyramid_rejected_reason"],  # type: ignore[arg-type]
    )


def _modify_sl_action(session: LivePyramidSession, sl: float, message: str) -> BridgeAction:
    action = _base_action(session, "PYRAMID_MODIFY_SL_ALL", message)
    return BridgeAction(
        action=action.action,
        trade_id=action.trade_id,
        pyramid_group_id=action.pyramid_group_id,
        setup_type=action.setup_type,
        symbol=action.symbol,
        direction=action.direction,
        message=message,
        sl=round(sl, 5),
        tp=round(session.mgr.take_profit, 5),
        ws_kalman_velocity=action.ws_kalman_velocity,
        ws_decel_exit=action.ws_decel_exit,
        ws_time_limit_exit=action.ws_time_limit_exit,
        ws_pyramid_rejected_reason=action.ws_pyramid_rejected_reason,
    )


def _cap_layer_lot(session: LivePyramidSession, lot: float, reference_price: float, sl: float) -> float:
    if session.equity <= 0.0:
        return lot
    return cap_pyramid_lot_size(
        lot,
        symbol=session.symbol,
        sl_price=sl,
        reference_price=reference_price,
        equity=session.equity,
        phase_start_equity=session.phase_start_equity or session.equity,
    )


def _limit_action(session: LivePyramidSession, pending: PendingLimitState, message: str) -> BridgeAction:
    action = _base_action(session, "PYRAMID_LIMIT", message)
    return BridgeAction(
        action=action.action,
        trade_id=action.trade_id,
        pyramid_group_id=action.pyramid_group_id,
        setup_type=action.setup_type,
        symbol=action.symbol,
        direction=action.direction,
        message=message,
        limit_price=pending.limit_price,
        lot_size=pending.lot_size,
        sl=pending.unified_sl,
        tp=round(session.mgr.take_profit, 5),
        layer_index=pending.layer_index,
        ttl_bars=pending.ttl_bars,
        pending_order_ticket=pending.order_ticket,
        ws_kalman_velocity=action.ws_kalman_velocity,
        ws_decel_exit=action.ws_decel_exit,
        ws_time_limit_exit=action.ws_time_limit_exit,
        ws_pyramid_rejected_reason=action.ws_pyramid_rejected_reason,
    )


def _cancel_action(session: LivePyramidSession, pending: PendingLimitState, message: str) -> BridgeAction:
    action = _base_action(session, "PYRAMID_CANCEL", message)
    return BridgeAction(
        action=action.action,
        trade_id=action.trade_id,
        pyramid_group_id=action.pyramid_group_id,
        setup_type=action.setup_type,
        symbol=action.symbol,
        direction=action.direction,
        message=message,
        pending_order_ticket=pending.order_ticket,
        layer_index=pending.layer_index,
        ws_kalman_velocity=action.ws_kalman_velocity,
        ws_decel_exit=action.ws_decel_exit,
        ws_time_limit_exit=action.ws_time_limit_exit,
        ws_pyramid_rejected_reason=action.ws_pyramid_rejected_reason,
    )


def _evaluate_ws_gates(session: LivePyramidSession, ws: WyckoffGateInput | None) -> WyckoffGateInput:
    if ws is not None:
        return ws
    return WyckoffGateInput()


def _maybe_move_sl_to_breakeven(session: LivePyramidSession, close: float) -> BridgeAction | None:
    mgr = session.mgr
    # Align with can_add_pyramid: SL→BE once unrealized >= 1R (not only at trigger_r).
    if mgr.portfolio_unrealized_r(close) >= 1.0 and not mgr._all_sl_at_breakeven():
        mgr.move_all_sl_to_breakeven()
        return _modify_sl_action(session, mgr.unified_stop_loss(), "move all SL to breakeven")
    return None


def _pending_ttl_expired(session: LivePyramidSession, pending: PendingLimitState) -> bool:
    bars_waiting = session.current_bar_index - pending.placed_at_bar_index
    return bars_waiting >= pending.ttl_bars


def evaluate_pyramid_on_bar(
    session: LivePyramidSession,
    bar: BarSnapshot,
    *,
    bar_index: int,
    daily_dd_remaining: float | None = None,
    ws_gates: WyckoffGateInput | None = None,
    ttl_bars: int | None = None,
) -> list[BridgeAction]:
    """
    確定バーごとのライブピラミッド評価。

    L5 BT の bar ループに相当。積み増しは PYRAMID_LIMIT として返し、即時 add_pyramid_layer は行わない。
    """
    if session.closed:
        return []

    session.current_bar_index = bar_index
    mgr = session.mgr
    actions: list[BridgeAction] = []
    effective_ttl = ttl_bars if ttl_bars is not None else LIVE_PYRAMID_LIMIT_TTL_BARS
    gates = _evaluate_ws_gates(session, ws_gates)

    sl_before_peak = mgr.unified_stop_loss()
    mgr.update_peak(bar.high, bar.low, bar.close)

    be_action = _maybe_move_sl_to_breakeven(session, bar.close)
    if be_action is not None:
        actions.append(be_action)
    elif mgr.unified_stop_loss() != sl_before_peak:
        actions.append(
            _modify_sl_action(session, mgr.unified_stop_loss(), "trailing SL ratchet sync")
        )

    if session.pending_limit is not None:
        pending = session.pending_limit
        if _pending_ttl_expired(session, pending):
            actions.append(_cancel_action(session, pending, "pending limit TTL expired"))
            session.pending_limit = None
            if LIVE_PYRAMID_USE_MARKET_FALLBACK:
                can_add, reason = mgr.can_add_pyramid(bar.close, daily_dd_remaining)
                if can_add and not gates.past_time_limit:
                    lot = mgr.pyramid_lot_for_next_layer()
                    sl = round(mgr.unified_stop_loss(), 5)
                    lot = _cap_layer_lot(session, lot, bar.close, sl)
                    if lot <= 0.0:
                        session.last_rejected_reason = "FINTOKEI_LAYER_CAP"
                        return actions
                    mgr.add_pyramid_layer(bar.close)
                    actions.append(
                        BridgeAction(
                            action="PYRAMID_MARKET_FALLBACK",
                            trade_id=session.trade_id,
                            pyramid_group_id=session.pyramid_group_id,
                            setup_type=session.setup_type,
                            symbol=session.symbol,
                            direction=session.direction,
                            message="market fallback after limit TTL",
                            limit_price=round(bar.close, 5),
                            lot_size=round(lot, 4),
                            sl=sl,
                            tp=round(mgr.take_profit, 5),
                            layer_index=mgr.layer_count,
                        )
                    )
                    actions.append(_modify_sl_action(session, mgr.unified_stop_loss(), "sync SL after market fallback"))
                elif reason and reason != "OK":
                    session.last_rejected_reason = reason
        return actions

    if gates.past_time_limit:
        session.time_limit_triggered = True

    can_add, reason = mgr.can_add_pyramid(bar.close, daily_dd_remaining)
    ready_for_pyramid = (
        mgr._all_sl_at_breakeven()
        and mgr.portfolio_unrealized_r(bar.close) >= session.trigger_r
        and mgr.layer_count < mgr.max_pyramid_layers
    )

    if can_add:
        if gates.past_time_limit:
            can_add = False
            reason = "TIME_LIMIT"
            session.time_limit_triggered = True
        elif mgr.portfolio_unrealized_r(bar.close) < session.trigger_r:
            can_add = False
            reason = "UNREALIZED_R_BELOW_TRIGGER"
    elif gates.past_time_limit and ready_for_pyramid:
        session.time_limit_triggered = True

    if can_add:
        intent = build_pyramid_limit_intent(mgr, bar.close, ttl_bars=effective_ttl)
        capped_lot = _cap_layer_lot(session, intent.lot_size, intent.limit_price, intent.unified_sl)
        if capped_lot <= 0.0:
            session.last_rejected_reason = "FINTOKEI_LAYER_CAP"
            return actions
        if capped_lot != intent.lot_size:
            intent = PyramidLimitIntent(
                limit_price=intent.limit_price,
                lot_size=round(capped_lot, 4),
                unified_sl=intent.unified_sl,
                layer_index=intent.layer_index,
                ttl_bars=intent.ttl_bars,
            )
        pending = attach_pending_limit(session, intent)
        actions.append(_limit_action(session, pending, f"pyramid layer {pending.layer_index} limit at bar close"))
    elif reason and reason != "OK":
        if reason == "SL_NOT_AT_BREAKEVEN":
            session.last_rejected_reason = reason
        elif session.last_rejected_reason != "SL_NOT_AT_BREAKEVEN":
            session.last_rejected_reason = reason

    return actions


def on_limit_filled(
    session: LivePyramidSession,
    fill_price: float,
    *,
    position_ticket: int | None = None,
) -> list[BridgeAction]:
    """Limit 約定通知 — PyramidManager 状態を同期し SL 一括更新指令を返す。"""
    if session.closed:
        return []

    mgr = session.mgr
    pending = session.pending_limit
    session.pending_limit = None

    if pending is None:
        return []

    mgr.add_pyramid_layer(fill_price)
    if position_ticket is not None:
        session.position_tickets.append(position_ticket)

    return [
        _modify_sl_action(
            session,
            mgr.unified_stop_loss(),
            f"limit filled layer {mgr.layer_count} at {fill_price}",
        )
    ]


def close_live_pyramid_session(session: LivePyramidSession) -> list[BridgeAction]:
    """トレード終了 — 未約定 Limit があればキャンセル指令を返す。"""
    actions: list[BridgeAction] = []
    if session.pending_limit is not None:
        actions.append(
            _cancel_action(session, session.pending_limit, "session closed — cancel pending limit")
        )
        session.pending_limit = None
    session.closed = True
    return actions
