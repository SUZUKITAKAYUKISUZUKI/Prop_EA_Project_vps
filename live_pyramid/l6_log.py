"""Live pyramid L6 audit log — CSV append + session event rows."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live_pyramid.actions import BridgeAction
from live_pyramid.session import LivePyramidSession

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LIVE_PYRAMID_LOG_PATH = LOG_DIR / "live_pyramid_audit_log.csv"

LIVE_PYRAMID_LOG_COLUMNS = [
    "event_time",
    "event_type",
    "trade_id",
    "pyramid_group_id",
    "setup_type",
    "symbol",
    "direction",
    "bar_index",
    "action_type",
    "limit_price",
    "lot_size",
    "sl",
    "tp",
    "layer_index",
    "pyramid_layers",
    "pending_limit",
    "fill_price",
    "order_ticket",
    "position_ticket",
    "limit_placed_count",
    "limit_filled_count",
    "limit_cancelled_count",
    "market_fallback_count",
    "ws_kalman_velocity",
    "ws_decel_exit",
    "ws_time_limit_exit",
    "ws_pyramid_rejected_reason",
    "pyramid_entry_prices",
    "message",
]


def live_pyramid_log_path() -> Path:
    raw = __import__("os").environ.get("LIVE_PYRAMID_LOG_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_LIVE_PYRAMID_LOG_PATH


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _session_stats(session: LivePyramidSession) -> dict[str, int]:
    return {
        "limit_placed_count": session.stats_limit_placed,
        "limit_filled_count": session.stats_limit_filled,
        "limit_cancelled_count": session.stats_limit_cancelled,
        "market_fallback_count": session.stats_market_fallback,
    }


def _session_ws_fields(session: LivePyramidSession) -> dict[str, Any]:
    return {
        "ws_kalman_velocity": round(session.kalman_velocity_at_entry, 8),
        "ws_decel_exit": session.decel_exit_triggered,
        "ws_time_limit_exit": session.time_limit_triggered,
        "ws_pyramid_rejected_reason": session.last_rejected_reason or "",
    }


def build_pyramid_log_row(
    session: LivePyramidSession,
    *,
    event_type: str,
    action_type: str = "",
    bar_index: int | None = None,
    limit_price: float | None = None,
    lot_size: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    layer_index: int | None = None,
    fill_price: float | None = None,
    order_ticket: int | None = None,
    position_ticket: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    mgr = session.mgr
    row: dict[str, Any] = {
        "event_time": _utc_now_str(),
        "event_type": event_type,
        "trade_id": session.trade_id,
        "pyramid_group_id": session.pyramid_group_id,
        "setup_type": session.setup_type,
        "symbol": session.symbol,
        "direction": session.direction,
        "bar_index": bar_index if bar_index is not None else session.current_bar_index,
        "action_type": action_type,
        "limit_price": limit_price,
        "lot_size": lot_size,
        "sl": sl,
        "tp": tp,
        "layer_index": layer_index,
        "pyramid_layers": mgr.pyramid_layers,
        "pending_limit": session.pending_limit is not None,
        "fill_price": fill_price,
        "order_ticket": order_ticket,
        "position_ticket": position_ticket,
        "pyramid_entry_prices": json.dumps(
            [round(p.entry_price, 5) for p in mgr.positions],
            ensure_ascii=False,
        ),
        "message": message,
        **_session_stats(session),
        **_session_ws_fields(session),
    }
    return row


def update_session_stats_from_action(session: LivePyramidSession, action: BridgeAction) -> None:
    if action.action == "PYRAMID_LIMIT":
        session.stats_limit_placed += 1
    elif action.action == "PYRAMID_CANCEL":
        session.stats_limit_cancelled += 1
    elif action.action == "PYRAMID_MARKET_FALLBACK":
        session.stats_market_fallback += 1


def rows_from_bridge_actions(
    session: LivePyramidSession,
    actions: list[BridgeAction],
    *,
    event_type: str,
    bar_index: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in actions:
        update_session_stats_from_action(session, action)
        rows.append(
            build_pyramid_log_row(
                session,
                event_type=event_type,
                action_type=action.action,
                bar_index=bar_index,
                limit_price=action.limit_price,
                lot_size=action.lot_size,
                sl=action.sl,
                tp=action.tp,
                layer_index=action.layer_index,
                order_ticket=action.pending_order_ticket,
                message=action.message,
            )
        )
    return rows


def append_live_pyramid_log_rows(
    rows: list[dict[str, Any]],
    *,
    log_path: Path | None = None,
) -> Path:
    if not rows:
        path = log_path or live_pyramid_log_path()
        return path

    path = log_path or live_pyramid_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIVE_PYRAMID_LOG_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in LIVE_PYRAMID_LOG_COLUMNS})
    return path


def log_pyramid_register(session: LivePyramidSession, *, message: str = "session registered") -> Path:
    row = build_pyramid_log_row(session, event_type="REGISTER", message=message)
    return append_live_pyramid_log_rows([row])


def log_pyramid_tick(
    session: LivePyramidSession,
    actions: list[BridgeAction],
    *,
    bar_index: int,
) -> Path:
    if not actions:
        row = build_pyramid_log_row(
            session,
            event_type="TICK",
            action_type="NOOP",
            bar_index=bar_index,
            message="no bridge actions",
        )
        return append_live_pyramid_log_rows([row])
    rows = rows_from_bridge_actions(session, actions, event_type="TICK", bar_index=bar_index)
    return append_live_pyramid_log_rows(rows)


def log_pyramid_fill(
    session: LivePyramidSession,
    actions: list[BridgeAction],
    *,
    fill_price: float,
    position_ticket: int | None = None,
    order_ticket: int | None = None,
) -> Path:
    session.stats_limit_filled += 1
    rows = rows_from_bridge_actions(session, actions, event_type="FILL")
    if rows:
        rows[0]["fill_price"] = round(fill_price, 5)
        rows[0]["position_ticket"] = position_ticket or ""
        rows[0]["order_ticket"] = order_ticket or rows[0].get("order_ticket", "")
        rows[0]["message"] = f"limit filled at {fill_price}"
    else:
        rows = [
            build_pyramid_log_row(
                session,
                event_type="FILL",
                action_type="PYRAMID_FILL",
                fill_price=fill_price,
                position_ticket=position_ticket,
                order_ticket=order_ticket,
                message=f"limit filled at {fill_price}",
            )
        ]
    return append_live_pyramid_log_rows(rows)


def log_pyramid_close(session: LivePyramidSession, actions: list[BridgeAction]) -> Path:
    rows = rows_from_bridge_actions(session, actions, event_type="CLOSE")
    summary = build_pyramid_log_row(
        session,
        event_type="SESSION_CLOSE",
        action_type="SUMMARY",
        message="session closed",
    )
    rows.append(summary)
    return append_live_pyramid_log_rows(rows)
