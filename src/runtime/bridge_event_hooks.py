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