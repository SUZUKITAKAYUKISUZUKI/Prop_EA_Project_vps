"""Bridge actions emitted by the live pyramid runtime for MT5 / EA execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BridgeActionType = Literal[
    "PYRAMID_LIMIT",
    "PYRAMID_CANCEL",
    "PYRAMID_MODIFY_SL_ALL",
    "PYRAMID_PARTIAL_CLOSE",
    "PYRAMID_MARKET_FALLBACK",
    "NOOP",
]


@dataclass(frozen=True)
class BridgeAction:
    action: BridgeActionType
    trade_id: str
    pyramid_group_id: str
    setup_type: str
    symbol: str
    direction: str
    message: str = ""
    limit_price: float | None = None
    lot_size: float | None = None
    sl: float | None = None
    tp: float | None = None
    layer_index: int | None = None
    ttl_bars: int | None = None
    position_tickets: list[int] = field(default_factory=list)
    pending_order_ticket: int | None = None
    ws_kalman_velocity: float | None = None
    ws_decel_exit: bool | None = None
    ws_time_limit_exit: bool | None = None
    ws_pyramid_rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["position_tickets"]:
            payload.pop("position_tickets")
        for key in list(payload.keys()):
            if payload[key] is None:
                payload.pop(key)
        return payload
