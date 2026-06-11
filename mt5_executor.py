"""
mt5_executor.py — L4.5 リスク予算に基づく MT5 成行発注・決済

Python ターミナルから MetaTrader5 パッケージ経由で直接発注する場合に使用。
EA (PropEA_Bridge.mq5) 経由の WebRequest 運用と併用可能。

前提:
  - MT5 ターミナルが起動・ログイン済み
  - pip install MetaTrader5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from feature_engineering import lot_from_risk_budget

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore[assignment]

DEFAULT_MAGIC = 20260601
DEFAULT_DEVIATION = 20


@dataclass
class TradeExecutionResult:
    success: bool
    action: str
    symbol: str
    lot: float
    ticket: int
    risk_budget: float
    message: str
    retcode: int | None = None


def ensure_mt5_initialized(path: str | None = None) -> bool:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed. Run: pip install MetaTrader5")
    if mt5.terminal_info() is not None:
        return True
    return bool(mt5.initialize(path=path) if path else mt5.initialize())


def normalize_lot(symbol: str, lot: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Symbol not found: {symbol}")
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    step = info.volume_step or 0.01
    lot = max(info.volume_min, min(info.volume_max, lot))
    steps = int(lot / step)
    return round(steps * step, 8)


def lot_from_risk_budget_mt5(
    symbol: str,
    risk_budget: float,
    entry: float,
    sl: float,
) -> float:
    """
    ブローカー tick_value / tick_size を用いて L4.5 risk_budget からロット逆算。
    """
    if mt5 is None:
        sl_distance = abs(entry - sl)
        return lot_from_risk_budget(risk_budget, sl_distance)

    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Symbol not found: {symbol}")

    sl_distance = abs(entry - sl)
    if sl_distance <= 0 or risk_budget <= 0:
        return 0.0

    tick_size = info.trade_tick_size or info.point
    tick_value = info.trade_tick_value
    if tick_size <= 0 or tick_value <= 0:
        return lot_from_risk_budget(risk_budget, sl_distance)

    ticks_at_risk = sl_distance / tick_size
    loss_per_lot = ticks_at_risk * tick_value
    if loss_per_lot <= 0:
        return 0.0
    return normalize_lot(symbol, risk_budget / loss_per_lot)


def _resolve_filling_mode(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_FOK
    filling = info.filling_mode
    if filling & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if filling & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def execute_trade(
    symbol: str,
    action: str,
    risk_budget: float,
    entry: float,
    sl: float,
    tp: float,
    magic: int = DEFAULT_MAGIC,
    comment: str = "PropEA_L45",
    deviation: int = DEFAULT_DEVIATION,
    mt5_path: str | None = None,
    lot_override: float | None = None,
) -> TradeExecutionResult:
    """
    L4.5 risk_budget からロットを逆算し、MT5 へ成行エントリーを送信。

    action: BUY / SELL
    """
    action = action.upper()
    if action not in ("BUY", "SELL"):
        return TradeExecutionResult(
            success=False,
            action=action,
            symbol=symbol,
            lot=0.0,
            ticket=0,
            risk_budget=risk_budget,
            message=f"Unsupported action: {action}",
        )

    ensure_mt5_initialized(mt5_path)
    if not mt5.symbol_select(symbol, True):
        return TradeExecutionResult(
            success=False,
            action=action,
            symbol=symbol,
            lot=0.0,
            ticket=0,
            risk_budget=risk_budget,
            message=f"Failed to select symbol: {symbol}",
        )

    lot = lot_override if lot_override and lot_override > 0 else lot_from_risk_budget_mt5(
        symbol, risk_budget, entry, sl
    )
    if lot <= 0:
        return TradeExecutionResult(
            success=False,
            action=action,
            symbol=symbol,
            lot=0.0,
            ticket=0,
            risk_budget=risk_budget,
            message="Calculated lot is zero",
        )

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return TradeExecutionResult(
            success=False,
            action=action,
            symbol=symbol,
            lot=lot,
            ticket=0,
            risk_budget=risk_budget,
            message="No tick data",
        )

    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if action == "BUY" else tick.bid

    request: dict[str, Any] = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _resolve_filling_mode(symbol),
    }

    result = mt5.order_send(request)
    if result is None:
        return TradeExecutionResult(
            success=False,
            action=action,
            symbol=symbol,
            lot=lot,
            ticket=0,
            risk_budget=risk_budget,
            message=f"order_send returned None: {mt5.last_error()}",
        )

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return TradeExecutionResult(
        success=success,
        action=action,
        symbol=symbol,
        lot=lot,
        ticket=int(result.order or result.deal or 0),
        risk_budget=risk_budget,
        message=result.comment or ("OK" if success else "Order failed"),
        retcode=int(result.retcode),
    )


def close_trade(
    symbol: str,
    magic: int = DEFAULT_MAGIC,
    deviation: int = DEFAULT_DEVIATION,
    mt5_path: str | None = None,
) -> TradeExecutionResult:
    """指定シンボル・マジックの保有ポジションを成行決済。"""
    ensure_mt5_initialized(mt5_path)
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return TradeExecutionResult(
            success=True,
            action="CLOSE",
            symbol=symbol,
            lot=0.0,
            ticket=0,
            risk_budget=0.0,
            message="No open position",
        )

    target = None
    for pos in positions:
        if int(pos.magic) == magic:
            target = pos
            break
    if target is None:
        return TradeExecutionResult(
            success=False,
            action="CLOSE",
            symbol=symbol,
            lot=0.0,
            ticket=0,
            risk_budget=0.0,
            message=f"No position with magic={magic}",
        )

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return TradeExecutionResult(
            success=False,
            action="CLOSE",
            symbol=symbol,
            lot=float(target.volume),
            ticket=int(target.ticket),
            risk_budget=0.0,
            message="No tick data",
        )

    if target.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        action = "CLOSE_SELL"
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        action = "CLOSE_BUY"

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(target.volume),
        "type": order_type,
        "position": int(target.ticket),
        "price": price,
        "deviation": deviation,
        "magic": magic,
        "comment": "PropEA_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _resolve_filling_mode(symbol),
    }

    result = mt5.order_send(request)
    if result is None:
        return TradeExecutionResult(
            success=False,
            action=action,
            symbol=symbol,
            lot=float(target.volume),
            ticket=int(target.ticket),
            risk_budget=0.0,
            message=f"close order_send None: {mt5.last_error()}",
        )

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return TradeExecutionResult(
        success=success,
        action=action,
        symbol=symbol,
        lot=float(target.volume),
        ticket=int(result.order or result.deal or target.ticket),
        risk_budget=0.0,
        message=result.comment or ("Closed" if success else "Close failed"),
        retcode=int(result.retcode),
    )


def execute_trade_from_signal(
    broker_symbol: str,
    signal: dict[str, Any],
    magic: int = DEFAULT_MAGIC,
    mt5_path: str | None = None,
) -> TradeExecutionResult:
    """evaluate_trade_signal / pending_to_trade_signal の dict から発注。"""
    action = str(signal.get("action", "HOLD")).upper()
    if action in ("HOLD", "REJECT"):
        return TradeExecutionResult(
            success=False,
            action=action,
            symbol=broker_symbol,
            lot=0.0,
            ticket=0,
            risk_budget=float(signal.get("risk_budget", 0.0)),
            message=str(signal.get("message", "No trade")),
        )

    risk_budget = float(signal.get("risk_budget", 0.0))
    lot_override = float(signal.get("lot_size", 0.0))

    result = execute_trade(
        symbol=broker_symbol,
        action=action,
        risk_budget=risk_budget,
        entry=float(signal.get("entry", 0.0)),
        sl=float(signal.get("sl", 0.0)),
        tp=float(signal.get("tp", 0.0)),
        magic=magic,
        comment=str(signal.get("trade_id", "PropEA_L45")),
        mt5_path=mt5_path,
        lot_override=lot_override if risk_budget <= 0 else None,
    )
    if result.success and signal.get("setup_type") == "CSPA" and signal.get("exit_mode") == "CSPA_BE_TRAIL":
        register_cspa_exit_tracker(broker_symbol, magic, signal)
    return result


# --- CSPA live exit management (Python MetaTrader5 直結運用) ---

_cspa_live_trackers: dict[int, Any] = {}
_cspa_last_bar_time: dict[int, int] = {}


def _normalize_price(symbol: str, price: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return price
    digits = info.digits
    return round(price, digits)


def modify_position_sl(
    ticket: int,
    symbol: str,
    new_sl: float,
    tp: float | None = None,
    magic: int = DEFAULT_MAGIC,
) -> TradeExecutionResult:
    """保有ポジションの SL/TP を変更。"""
    ensure_mt5_initialized()
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return TradeExecutionResult(
            success=False,
            action="MODIFY_SL",
            symbol=symbol,
            lot=0.0,
            ticket=ticket,
            risk_budget=0.0,
            message="Position not found",
        )
    pos = positions[0]
    if int(pos.magic) != magic:
        return TradeExecutionResult(
            success=False,
            action="MODIFY_SL",
            symbol=symbol,
            lot=float(pos.volume),
            ticket=ticket,
            risk_budget=0.0,
            message=f"Magic mismatch: {pos.magic}",
        )

    norm_sl = _normalize_price(symbol, new_sl)
    norm_tp = _normalize_price(symbol, float(tp if tp is not None else pos.tp))
    cur_sl = float(pos.sl)
    point = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.00001
    if abs(norm_sl - cur_sl) < point * 0.5:
        return TradeExecutionResult(
            success=True,
            action="MODIFY_SL",
            symbol=symbol,
            lot=float(pos.volume),
            ticket=ticket,
            risk_budget=0.0,
            message="SL unchanged",
        )

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": norm_sl,
        "tp": norm_tp,
        "magic": magic,
    }
    result = mt5.order_send(request)
    if result is None:
        return TradeExecutionResult(
            success=False,
            action="MODIFY_SL",
            symbol=symbol,
            lot=float(pos.volume),
            ticket=ticket,
            risk_budget=0.0,
            message=f"modify SL failed: {mt5.last_error()}",
        )
    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return TradeExecutionResult(
        success=success,
        action="MODIFY_SL",
        symbol=symbol,
        lot=float(pos.volume),
        ticket=ticket,
        risk_budget=0.0,
        message=result.comment or ("OK" if success else "Modify failed"),
        retcode=int(result.retcode),
    )


def register_cspa_exit_tracker(symbol: str, magic: int, signal: dict[str, Any]) -> None:
    """直近エントリーの CSPA ポジションに出口トラッカーを紐付け。"""
    from strategies.archive.cspa_exit import CspaExitTracker

    ensure_mt5_initialized()
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    target = None
    for pos in positions:
        if int(pos.magic) == magic:
            target = pos
            break
    if target is None:
        return

    direction = "BUY" if target.type == mt5.POSITION_TYPE_BUY else "SELL"
    tracker = CspaExitTracker.from_signal(
        direction=direction,
        entry=float(target.price_open),
        initial_sl=float(signal.get("sl", target.sl)),
        take_profit=float(signal.get("tp", target.tp)),
        exit_fields=signal,
    )
    _cspa_live_trackers[int(target.ticket)] = tracker


def manage_cspa_exits(
    symbol: str,
    *,
    magic: int = DEFAULT_MAGIC,
    timeframe: int | None = None,
    mt5_path: str | None = None,
) -> list[TradeExecutionResult]:
    """確定 M5 バーごとに CSPA 建値 / トレール SL を更新（EA と同一ルール）。"""
    if mt5 is None:
        return []
    ensure_mt5_initialized(mt5_path)
    tf = timeframe if timeframe is not None else mt5.TIMEFRAME_M5
    rates = mt5.copy_rates_from_pos(symbol, tf, 1, 1)
    if rates is None or len(rates) == 0:
        return []

    bar = rates[0]
    bar_time = int(bar["time"])
    results: list[TradeExecutionResult] = []

    stale = [ticket for ticket in _cspa_live_trackers if not mt5.positions_get(ticket=ticket)]
    for ticket in stale:
        _cspa_live_trackers.pop(ticket, None)
        _cspa_last_bar_time.pop(ticket, None)

    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return results

    for pos in positions:
        if int(pos.magic) != magic:
            continue
        ticket = int(pos.ticket)
        tracker = _cspa_live_trackers.get(ticket)
        if tracker is None:
            continue
        if _cspa_last_bar_time.get(ticket) == bar_time:
            continue
        _cspa_last_bar_time[ticket] = bar_time

        new_sl = tracker.on_bar(float(bar["high"]), float(bar["low"]), float(bar["close"]))
        mod = modify_position_sl(ticket, symbol, new_sl, tp=float(pos.tp), magic=magic)
        results.append(mod)

    return results


# --- Live pyramid Limit-order execution (Python MetaTrader5 直結運用) ---


def place_pyramid_limit(
    symbol: str,
    direction: str,
    limit_price: float,
    lot: float,
    sl: float,
    tp: float,
    *,
    magic: int = DEFAULT_MAGIC,
    comment: str = "PropEA_PYR",
    deviation: int = DEFAULT_DEVIATION,
    mt5_path: str | None = None,
) -> TradeExecutionResult:
    """ピラミッド積み増し — Buy/Sell Limit 指値発注。"""
    direction = direction.upper()
    if direction not in ("BUY", "SELL"):
        return TradeExecutionResult(
            success=False,
            action="PYRAMID_LIMIT",
            symbol=symbol,
            lot=0.0,
            ticket=0,
            risk_budget=0.0,
            message=f"Unsupported direction: {direction}",
        )
    if lot <= 0 or limit_price <= 0:
        return TradeExecutionResult(
            success=False,
            action="PYRAMID_LIMIT",
            symbol=symbol,
            lot=lot,
            ticket=0,
            risk_budget=0.0,
            message="Invalid lot or limit price",
        )

    ensure_mt5_initialized(mt5_path)
    if not mt5.symbol_select(symbol, True):
        return TradeExecutionResult(
            success=False,
            action="PYRAMID_LIMIT",
            symbol=symbol,
            lot=lot,
            ticket=0,
            risk_budget=0.0,
            message=f"Failed to select symbol: {symbol}",
        )

    lot = normalize_lot(symbol, lot)
    norm_price = _normalize_price(symbol, limit_price)
    norm_sl = _normalize_price(symbol, sl)
    norm_tp = _normalize_price(symbol, tp)
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT

    request: dict[str, Any] = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": norm_price,
        "sl": norm_sl,
        "tp": norm_tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _resolve_filling_mode(symbol),
    }
    result = mt5.order_send(request)
    if result is None:
        return TradeExecutionResult(
            success=False,
            action="PYRAMID_LIMIT",
            symbol=symbol,
            lot=lot,
            ticket=0,
            risk_budget=0.0,
            message=f"limit order_send None: {mt5.last_error()}",
        )
    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return TradeExecutionResult(
        success=success,
        action="PYRAMID_LIMIT",
        symbol=symbol,
        lot=lot,
        ticket=int(result.order or 0),
        risk_budget=0.0,
        message=result.comment or ("OK" if success else "Limit order failed"),
        retcode=int(result.retcode),
    )


def cancel_pyramid_pending_order(
    order_ticket: int,
    symbol: str,
    *,
    mt5_path: str | None = None,
) -> TradeExecutionResult:
    """未約定ピラミッド Limit をキャンセル。"""
    ensure_mt5_initialized(mt5_path)
    if order_ticket <= 0:
        return TradeExecutionResult(
            success=True,
            action="PYRAMID_CANCEL",
            symbol=symbol,
            lot=0.0,
            ticket=0,
            risk_budget=0.0,
            message="No pending order ticket",
        )

    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": order_ticket,
        "symbol": symbol,
    }
    result = mt5.order_send(request)
    if result is None:
        return TradeExecutionResult(
            success=False,
            action="PYRAMID_CANCEL",
            symbol=symbol,
            lot=0.0,
            ticket=order_ticket,
            risk_budget=0.0,
            message=f"cancel order_send None: {mt5.last_error()}",
        )
    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return TradeExecutionResult(
        success=success,
        action="PYRAMID_CANCEL",
        symbol=symbol,
        lot=0.0,
        ticket=order_ticket,
        risk_budget=0.0,
        message=result.comment or ("Cancelled" if success else "Cancel failed"),
        retcode=int(result.retcode),
    )


def modify_pyramid_group_sl(
    symbol: str,
    new_sl: float,
    tp: float | None = None,
    *,
    magic: int = DEFAULT_MAGIC,
    pyramid_group_id: str | None = None,
    mt5_path: str | None = None,
) -> list[TradeExecutionResult]:
    """同一 pyramid_group_id（comment プレフィックス）の全ポジション SL を一括更新。"""
    from live_pyramid.mt5_dispatch import matches_pyramid_group

    ensure_mt5_initialized(mt5_path)
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return [
            TradeExecutionResult(
                success=True,
                action="PYRAMID_MODIFY_SL_ALL",
                symbol=symbol,
                lot=0.0,
                ticket=0,
                risk_budget=0.0,
                message="No open positions",
            )
        ]

    results: list[TradeExecutionResult] = []
    for pos in positions:
        if int(pos.magic) != magic:
            continue
        comment = str(pos.comment or "")
        if pyramid_group_id:
            if not matches_pyramid_group(comment, pyramid_group_id) and not comment.startswith("PropEA"):
                continue
        elif not comment.startswith("PropEA"):
            continue
        mod = modify_position_sl(
            int(pos.ticket),
            symbol,
            new_sl,
            tp=float(tp if tp is not None else pos.tp),
            magic=magic,
        )
        results.append(
            TradeExecutionResult(
                success=mod.success,
                action="PYRAMID_MODIFY_SL_ALL",
                symbol=mod.symbol,
                lot=mod.lot,
                ticket=mod.ticket,
                risk_budget=0.0,
                message=mod.message,
                retcode=mod.retcode,
            )
        )
    return results


def close_pyramid_positions_by_ticket(
    tickets: list[int],
    symbol: str,
    *,
    magic: int = DEFAULT_MAGIC,
    deviation: int = DEFAULT_DEVIATION,
    mt5_path: str | None = None,
) -> list[TradeExecutionResult]:
    """指定 ticket のポジションのみ成行決済（減速部分決済用）。"""
    ensure_mt5_initialized(mt5_path)
    results: list[TradeExecutionResult] = []
    for ticket in tickets:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            continue
        pos = positions[0]
        if int(pos.magic) != magic or str(pos.symbol) != symbol:
            continue

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            results.append(
                TradeExecutionResult(
                    success=False,
                    action="PYRAMID_PARTIAL_CLOSE",
                    symbol=symbol,
                    lot=float(pos.volume),
                    ticket=ticket,
                    risk_budget=0.0,
                    message="No tick data",
                )
            )
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos.volume),
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": magic,
            "comment": "PropEA_PYR_partial",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _resolve_filling_mode(symbol),
        }
        result = mt5.order_send(request)
        if result is None:
            results.append(
                TradeExecutionResult(
                    success=False,
                    action="PYRAMID_PARTIAL_CLOSE",
                    symbol=symbol,
                    lot=float(pos.volume),
                    ticket=ticket,
                    risk_budget=0.0,
                    message=f"partial close None: {mt5.last_error()}",
                )
            )
            continue
        success = result.retcode == mt5.TRADE_RETCODE_DONE
        results.append(
            TradeExecutionResult(
                success=success,
                action="PYRAMID_PARTIAL_CLOSE",
                symbol=symbol,
                lot=float(pos.volume),
                ticket=ticket,
                risk_budget=0.0,
                message=result.comment or ("Closed" if success else "Partial close failed"),
                retcode=int(result.retcode),
            )
        )
    return results


def execute_pyramid_bridge_action(
    action: dict[str, Any],
    *,
    magic: int = DEFAULT_MAGIC,
    mt5_path: str | None = None,
) -> TradeExecutionResult | list[TradeExecutionResult]:
    """live_pyramid BridgeAction dict を MT5 操作へ変換・実行。"""
    from live_pyramid.mt5_dispatch import bridge_action_order_spec

    spec = bridge_action_order_spec(action)
    kind = spec["kind"]
    symbol = spec["symbol"]

    if kind == "NOOP" or not symbol:
        return TradeExecutionResult(
            success=True,
            action=str(action.get("action", "NOOP")),
            symbol=symbol or "",
            lot=0.0,
            ticket=0,
            risk_budget=0.0,
            message=spec["message"] or "noop",
        )

    if kind == "PLACE_LIMIT":
        return place_pyramid_limit(
            symbol=symbol,
            direction=spec["direction"],
            limit_price=spec["limit_price"],
            lot=spec["lot_size"],
            sl=spec["sl"],
            tp=spec["tp"],
            magic=magic,
            comment=spec["comment"],
            mt5_path=mt5_path,
        )

    if kind == "CANCEL_PENDING":
        return cancel_pyramid_pending_order(
            spec["pending_order_ticket"],
            symbol,
            mt5_path=mt5_path,
        )

    if kind == "MODIFY_GROUP_SL":
        return modify_pyramid_group_sl(
            symbol,
            spec["sl"],
            tp=spec["tp"] if spec["tp"] > 0 else None,
            magic=magic,
            pyramid_group_id=spec["pyramid_group_id"] or None,
            mt5_path=mt5_path,
        )

    if kind == "CLOSE_TICKETS":
        return close_pyramid_positions_by_ticket(
            spec["position_tickets"],
            symbol,
            magic=magic,
            mt5_path=mt5_path,
        )

    if kind == "MARKET_ADD":
        return execute_trade(
            symbol=symbol,
            action=spec["direction"],
            risk_budget=0.0,
            entry=spec["limit_price"],
            sl=spec["sl"],
            tp=spec["tp"],
            magic=magic,
            comment=spec["comment"],
            mt5_path=mt5_path,
            lot_override=spec["lot_size"],
        )

    return TradeExecutionResult(
        success=False,
        action=str(action.get("action", "")),
        symbol=symbol,
        lot=0.0,
        ticket=0,
        risk_budget=0.0,
        message=f"Unsupported bridge action kind: {kind}",
    )


def execute_pyramid_bridge_actions(
    actions: list[dict[str, Any]],
    *,
    magic: int = DEFAULT_MAGIC,
    mt5_path: str | None = None,
) -> list[TradeExecutionResult]:
    """BridgeAction リストを順次実行。"""
    results: list[TradeExecutionResult] = []
    for action in actions:
        outcome = execute_pyramid_bridge_action(action, magic=magic, mt5_path=mt5_path)
        if isinstance(outcome, list):
            results.extend(outcome)
        else:
            results.append(outcome)
    return results
