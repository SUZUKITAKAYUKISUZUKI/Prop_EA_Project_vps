"""Map BT/WFT runner record dicts to native SQLite row shapes."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.database.data_source import FEATURE_LOG_SCHEMA_VERSION, PORTFOLIO_DB_SCHEMA_VERSION


def _as_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def record_to_bt_trade(record: dict[str, Any], *, run_id: str) -> dict[str, Any] | None:
    trade_id = _as_str(record.get("trade_id"))
    if not trade_id:
        return None
    open_time = _as_str(record.get("timestamp") or record.get("entry_time"))
    holding = _as_float(record.get("holding_time"))
    close_time = open_time
    if open_time and holding is not None:
        try:
            close_time = (pd.to_datetime(open_time) + pd.to_timedelta(holding, unit="m")).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            close_time = open_time
    r_val = _as_float(record.get("profit_r", record.get("sized_result_r")))
    return {
        "trade_id": trade_id,
        "run_id": run_id,
        "strategy": _as_str(record.get("setup_type") or record.get("strategy")),
        "symbol": _as_str(record.get("pair") or record.get("symbol")),
        "open_time": open_time,
        "close_time": close_time,
        "direction": _as_str(record.get("direction") or record.get("side")),
        "r_multiple": r_val,
        "pnl": _as_float(record.get("profit_loss") or record.get("profit")),
        "exit_reason": _as_str(record.get("trade_result") or record.get("result")),
    }


def records_to_bt_trades(records: list[dict[str, Any]], *, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = record_to_bt_trade(record, run_id=run_id)
        if row is not None:
            rows.append(row)
    return rows


def compute_pf_from_records(records: list[dict[str, Any]]) -> float:
    executed = [r for r in records if r.get("trade_result") not in (None, "NOT_EXECUTED")]
    gross_profit = 0.0
    gross_loss = 0.0
    for record in executed:
        r_val = _as_float(record.get("profit_r", record.get("sized_result_r"))) or 0.0
        if r_val > 0:
            gross_profit += r_val
        elif r_val < 0:
            gross_loss += abs(r_val)
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_avg_r_from_records(records: list[dict[str, Any]]) -> float:
    executed = [r for r in records if r.get("trade_result") not in (None, "NOT_EXECUTED")]
    if not executed:
        return 0.0
    total = sum(_as_float(r.get("profit_r", r.get("sized_result_r"))) or 0.0 for r in executed)
    return total / len(executed)


def metrics_to_bt_run_summary(
    *,
    run_id: str,
    strategy: str | None,
    symbol: str | None,
    started_at: str,
    finished_at: str,
    metrics: Any,
    records: list[dict[str, Any]],
    source_version: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    executed = [r for r in records if r.get("trade_result") not in (None, "NOT_EXECUTED")]
    pf = compute_pf_from_records(records)
    return {
        "run_id": run_id,
        "strategy": strategy,
        "symbol": symbol,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_trades": int(getattr(metrics, "executed_trades", len(executed))),
        "total_r": float(getattr(metrics, "total_profit_r", 0.0)),
        "pf": None if pf == float("inf") else pf,
        "win_rate": float(getattr(metrics, "win_rate_pct", 0.0)),
        "avg_r": compute_avg_r_from_records(records),
        "max_dd": float(getattr(metrics, "max_total_dd_pct", 0.0)),
        "sharpe": None,
        "source_version": source_version or str(PORTFOLIO_DB_SCHEMA_VERSION),
        "notes": notes,
    }


def legacy_trade_row(record: dict[str, Any], *, run_id: int) -> dict[str, Any]:
    trade = record_to_bt_trade(record, run_id=str(run_id))
    if trade is None:
        return {}
    return {
        "strategy": trade["strategy"],
        "symbol": trade["symbol"],
        "direction": trade["direction"],
        "entry_time": trade["open_time"],
        "exit_time": trade["close_time"],
        "r_multiple": trade["r_multiple"],
        "profit": trade["pnl"],
        "result": trade["exit_reason"],
        "source_trade_id": trade["trade_id"],
    }


def dataframe_to_bt_records(
    df: pd.DataFrame,
    *,
    default_strategy: str | None = None,
    r_column: str | None = None,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records: list[dict[str, Any]] = []
    r_candidates = [c for c in (r_column, "profit_r", "result_r", "sized_result_r", "R") if c]
    for idx, row in df.iterrows():
        payload = {str(k): row[k] for k in df.columns}
        if not payload.get("trade_id"):
            payload["trade_id"] = f"row_{idx}"
        if not payload.get("timestamp"):
            for key in ("entry_time", "open_time", "datetime"):
                if payload.get(key):
                    payload["timestamp"] = payload[key]
                    break
        r_val = None
        for col in r_candidates:
            if col and col in payload and not pd.isna(payload[col]):
                r_val = payload[col]
                payload["profit_r"] = r_val
                break
        if payload.get("trade_result") in (None, "") and r_val is not None:
            try:
                payload["trade_result"] = "WIN" if float(r_val) > 0 else "LOSS"
            except (TypeError, ValueError):
                payload["trade_result"] = "WIN"
        if not payload.get("setup_type"):
            payload["setup_type"] = default_strategy or payload.get("strategy")
        if not payload.get("pair") and payload.get("symbol"):
            payload["pair"] = payload["symbol"]
        records.append(payload)
    return records
