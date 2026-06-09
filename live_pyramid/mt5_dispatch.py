"""Map BridgeAction dicts to MT5 execution (testable without MetaTrader5)."""

from __future__ import annotations

from typing import Any, Literal

OrderSpecKind = Literal[
    "PLACE_LIMIT",
    "CANCEL_PENDING",
    "MODIFY_GROUP_SL",
    "CLOSE_TICKETS",
    "MARKET_ADD",
    "NOOP",
]


def bridge_action_kind(action: dict[str, Any]) -> OrderSpecKind:
    name = str(action.get("action", "NOOP")).upper()
    mapping = {
        "PYRAMID_LIMIT": "PLACE_LIMIT",
        "PYRAMID_CANCEL": "CANCEL_PENDING",
        "PYRAMID_MODIFY_SL_ALL": "MODIFY_GROUP_SL",
        "PYRAMID_PARTIAL_CLOSE": "CLOSE_TICKETS",
        "PYRAMID_MARKET_FALLBACK": "MARKET_ADD",
        "NOOP": "NOOP",
    }
    return mapping.get(name, "NOOP")


def pyramid_comment(pyramid_group_id: str, layer_index: int) -> str:
    group = pyramid_group_id[:12] if pyramid_group_id else "pyr"
    return f"PropEA_PYR_{group}_L{layer_index}"


def matches_pyramid_group(comment: str, pyramid_group_id: str) -> bool:
    if not pyramid_group_id:
        return comment.startswith("PropEA_PYR_")
    prefix = f"PropEA_PYR_{pyramid_group_id[:12]}"
    return comment.startswith(prefix)


def bridge_action_order_spec(action: dict[str, Any]) -> dict[str, Any]:
    """Normalize BridgeAction JSON to a flat execution spec for mt5_executor."""
    kind = bridge_action_kind(action)
    group_id = str(action.get("pyramid_group_id", ""))
    layer_index = int(action.get("layer_index", 0) or 0)
    return {
        "kind": kind,
        "symbol": str(action.get("symbol", "")),
        "direction": str(action.get("direction", "")).upper(),
        "trade_id": str(action.get("trade_id", "")),
        "pyramid_group_id": group_id,
        "limit_price": float(action.get("limit_price", 0.0) or 0.0),
        "lot_size": float(action.get("lot_size", 0.0) or 0.0),
        "sl": float(action.get("sl", 0.0) or 0.0),
        "tp": float(action.get("tp", 0.0) or 0.0),
        "layer_index": layer_index,
        "comment": pyramid_comment(group_id, layer_index) if layer_index > 0 else f"PropEA_PYR_{group_id[:12]}",
        "pending_order_ticket": int(action.get("pending_order_ticket", 0) or 0),
        "position_tickets": [int(t) for t in action.get("position_tickets", []) if t],
        "message": str(action.get("message", "")),
    }
