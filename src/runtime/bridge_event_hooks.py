"""Bridge endpoint hooks — emit trade events without changing business logic."""
from __future__ import annotations

from typing import Any

from src.runtime.feature_snapshot import build_feature_snapshot
from src.runtime.trade_logger import get_trade_logger


def _logger():
    return get_trade_logger()


def log_trade_signal_result(
    request: Any,
    result: dict[str, Any],
    *,
    pending: Any | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    logger = _logger()
    if not logger.enabled:
        return
    market = getattr(request, "market", None)
    symbol = getattr(market, "pair", None) if market else result.get("symbol")
    logger.emit_trade_signal(
        action=str(result.get("action", "UNKNOWN")),
        trade_id=result.get("trade_id"),
        strategy=result.get("setup_type"),
        symbol=symbol,
        lot=result.get("lot_size"),
        sl=result.get("sl"),
        tp=result.get("tp"),
        entry=result.get("entry"),
        decision_source=result.get("decision_source"),
        lot_factor=result.get("lot_factor"),
        message=result.get("message"),
        sentinel_tags=result.get("sentinel_tags"),
    )

    if pending is not None:
        features = build_feature_snapshot(pending, payload=payload, signal=result)
        logger.emit_feature_snapshot(
            trade_id=str(getattr(pending, "trade_id", result.get("trade_id", ""))),
            strategy=str(getattr(pending, "setup_type", result.get("setup_type", "UNKNOWN"))),
            symbol=str(symbol or features.get("symbol") or "UNKNOWN"),
            features=features,
            action=result.get("action"),
            decision_source=getattr(pending, "decision_source", None),
        )

    if str(result.get("action", "")).upper() in {"ALLOW", "OPEN", "BUY", "SELL"} and result.get("trade_id"):
        open_extra: dict[str, Any] = {}
        if pending is not None:
            open_extra["candidate_score"] = getattr(pending, "candidate_score", None)
            open_extra["bayes_probability"] = getattr(pending, "bayes_probability", None)
        logger.emit_trade_open(
            trade_id=str(result["trade_id"]),
            strategy=str(result.get("setup_type") or "UNKNOWN"),
            symbol=str(symbol or "UNKNOWN"),
            direction=str(result.get("action", "UNKNOWN")),
            entry_price=float(result.get("entry") or getattr(market, "close", 0.0) or 0.0),
            sl=result.get("sl"),
            tp=result.get("tp"),
            lot=result.get("lot_size"),
            risk_r=result.get("lot_factor"),
            **open_extra,
        )


def log_sentinel_tick(verdict: Any, request: Any) -> None:
    logger = _logger()
    if not logger.enabled:
        return
    if getattr(verdict, "entry_allowed", True):
        return
    logger.emit_sentinel_block(
        reason=getattr(verdict, "message", "entry blocked"),
        tags=list(getattr(verdict, "tags", []) or []),
        panic_close=getattr(verdict, "panic_close", False),
        spread_block=getattr(verdict, "spread_block", False),
        rollover_block=getattr(verdict, "rollover_block", False),
        equity=getattr(request, "equity", None),
    )


def log_pyramid_register(request: Any, session: Any) -> None:
    logger = _logger()
    if not logger.enabled:
        return
    logger.emit_trade_open(
        trade_id=str(request.trade_id),
        strategy=str(request.setup_type),
        symbol=str(request.symbol),
        direction=str(request.direction),
        entry_price=float(request.entry),
        sl=float(request.sl),
        tp=float(request.tp),
        lot=float(request.lot_size),
        pyramid_group_id=getattr(session, "pyramid_group_id", None),
    )


def log_pyramid_actions(session: Any, actions: list[Any]) -> None:
    logger = _logger()
    if not logger.enabled:
        return
    for action in actions:
        payload = action.to_dict() if hasattr(action, "to_dict") else dict(action)
        act = payload.get("action", "")
        trade_id = str(payload.get("trade_id", getattr(session, "trade_id", "")))
        if act == "PYRAMID_LIMIT":
            logger.emit_pyramid_add(
                trade_id=trade_id,
                strategy=payload.get("setup_type"),
                symbol=payload.get("symbol"),
                direction=payload.get("direction"),
                limit_price=payload.get("limit_price"),
                lot=payload.get("lot_size"),
                layer_index=payload.get("layer_index"),
            )
        elif act == "PYRAMID_MODIFY_SL_ALL":
            if payload.get("sl") is not None:
                logger.emit_sl_modify(
                    trade_id=trade_id,
                    sl=float(payload["sl"]),
                    strategy=payload.get("setup_type"),
                    symbol=payload.get("symbol"),
                )
            if payload.get("tp") is not None:
                logger.emit_tp_modify(
                    trade_id=trade_id,
                    tp=float(payload["tp"]),
                    strategy=payload.get("setup_type"),
                    symbol=payload.get("symbol"),
                )


def log_pyramid_close(session: Any, actions: list[Any]) -> None:
    logger = _logger()
    if not logger.enabled:
        return
    log_pyramid_actions(session, actions)
    logger.emit_trade_close(
        trade_id=str(getattr(session, "trade_id", "")),
        strategy=str(getattr(session, "setup_type", None)),
        symbol=str(getattr(session, "symbol", None)),
        reason="pyramid_close",
    )


def log_dbbs_trade_closed(result_r: float, snapshot: dict[str, Any]) -> None:
    logger = _logger()
    if not logger.enabled:
        return
    logger.emit_trade_close(
        trade_id="DBBS",
        strategy="DBBS",
        profit_r=result_r,
        result="WIN" if result_r > 0 else "LOSS",
        bear_kill_switch_active=snapshot.get("bear_kill_switch_active"),
        last_3_avg_r=snapshot.get("last_3_avg_r"),
    )
    if snapshot.get("bear_kill_switch_active"):
        logger.emit_sentinel_block(reason="DBBS Bear Kill Switch active", strategy="DBBS")


def log_pet_decision(decision: Any) -> None:
    logger = _logger()
    if not logger.enabled or decision is None:
        return
    payload = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
    action = str(payload.get("action", "")).upper()
    if (
        action not in {"FORCE_CLOSE", "HALT", "EXIT", "CLOSE", "CLOSE_ALL", "CLOSE_LOWEST_RANKED"}
        and not payload.get("close_all")
    ):
        return
    logger.emit_pet_exit(
        trade_id=payload.get("trade_id"),
        strategy=payload.get("setup_type"),
        action=action,
        reason=payload.get("reason") or payload.get("message"),
        stage_name=payload.get("stage_name"),
        close_all=payload.get("close_all"),
        close_tickets=payload.get("close_tickets"),
    )
