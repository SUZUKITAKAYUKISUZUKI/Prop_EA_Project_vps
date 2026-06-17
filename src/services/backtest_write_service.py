"""Persist backtest results directly to portfolio_os.db."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.database.db_manager import utc_now_iso
from src.repositories.backtest_repository import BacktestRepository
from src.repositories.base import create_default_db_manager
from src.services.bt_wft_records import (
    legacy_trade_row,
    metrics_to_bt_run_summary,
    records_to_bt_trades,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _strategy_label(strategies: Any) -> str:
    if strategies is None:
        return "portfolio"
    if isinstance(strategies, (list, tuple)):
        return "+".join(str(s) for s in strategies)
    return str(strategies)


class BacktestWriteService:
    def __init__(
        self,
        *,
        backtest_repo: BacktestRepository | None = None,
        db=None,
    ) -> None:
        if backtest_repo is not None:
            self._bt = backtest_repo
            self._owns = False
        else:
            self._db = db or create_default_db_manager()
            self._bt = BacktestRepository(self._db, owns_connection=False)
            self._owns = db is None

    def close(self) -> None:
        if self._owns:
            self._bt.close()

    def save_bt_result(
        self,
        records: list[dict[str, Any]] | pd.DataFrame,
        summary: dict[str, Any] | Any | None = None,
        *,
        metrics: Any | None = None,
        strategy: str | None = None,
        symbol: str | None = None,
        run_id: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        notes: str | None = None,
        output_path: str | None = None,
        mirror_legacy: bool = True,
    ) -> str:
        if isinstance(records, pd.DataFrame):
            record_list = records.to_dict(orient="records")
        else:
            record_list = list(records)

        if metrics is None and summary is not None and hasattr(summary, "total_profit_r"):
            metrics = summary

        if metrics is None and record_list:
            from backtest_runner import compute_backtest_metrics

            metrics = compute_backtest_metrics(record_list)

        strategy_name = strategy or "portfolio"

        bt_run_id = run_id or f"bt_{strategy_name}_{uuid.uuid4().hex[:12]}"
        started = started_at or (
            str(record_list[0].get("timestamp"))[:19] if record_list else _utc_now()
        )
        finished = finished_at or (
            str(record_list[-1].get("timestamp"))[:19] if record_list else _utc_now()
        )

        run_row = metrics_to_bt_run_summary(
            run_id=bt_run_id,
            strategy=strategy_name,
            symbol=symbol,
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            records=record_list,
            notes=notes or output_path,
        )
        self._bt.save_run(run_row)
        trades = records_to_bt_trades(record_list, run_id=bt_run_id)
        self._bt.save_trades(bt_run_id, trades)

        if mirror_legacy:
            self._mirror_legacy_tables(
                bt_run_id=bt_run_id,
                strategy=strategy_name,
                records=record_list,
                metrics=metrics,
                output_path=output_path,
            )
        return bt_run_id

    def _mirror_legacy_tables(
        self,
        *,
        bt_run_id: str,
        strategy: str,
        records: list[dict[str, Any]],
        metrics: Any,
        output_path: str | None,
    ) -> None:
        db = self._bt._db
        description = output_path or bt_run_id
        legacy_run_id = db.insert_run(
            "bt",
            strategy=strategy,
            description=description,
            parameters={"bt_run_id": bt_run_id, "native": True},
            source="BACKTEST",
        )
        executed = [r for r in records if r.get("trade_result") not in (None, "NOT_EXECUTED")]
        for record in executed:
            payload = legacy_trade_row(record, run_id=legacy_run_id)
            if payload:
                db.insert_trade(legacy_run_id, **payload, source="BACKTEST")

        if metrics is not None:
            from src.services.bt_wft_records import compute_pf_from_records

            pf = compute_pf_from_records(records)
            db.insert_bt_summary(
                legacy_run_id,
                pf=None if pf == float("inf") else pf,
                wr=float(getattr(metrics, "win_rate_pct", 0.0)),
                total_r=float(getattr(metrics, "total_profit_r", 0.0)),
                max_dd=float(getattr(metrics, "max_total_dd_pct", 0.0)),
                label="aggregate",
            )

        if output_path:
            db.register_import(
                output_path.replace("\\", "/"),
                legacy_run_id,
                "trade",
                len(executed),
                None,
            )
        self._bt.register_legacy_run(
            run_id=bt_run_id,
            strategy=strategy,
            description=description,
            legacy_run_id=legacy_run_id,
        )


def save_backtest_run(
    records: list[dict[str, Any]] | pd.DataFrame,
    summary: dict[str, Any] | Any | None = None,
    **kwargs: Any,
) -> str:
    """Module-level helper used by BT runner hooks."""
    enabled = os.environ.get("BT_WRITE_SQLITE", "1").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        return ""
    service = BacktestWriteService()
    try:
        return service.save_bt_result(records, summary, **kwargs)
    finally:
        service.close()
