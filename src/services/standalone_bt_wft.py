"""Persist standalone pure-BT / feature-log pipelines to SQLite."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.database.db_manager import utc_now_iso
from src.repositories.backtest_repository import BacktestRepository
from src.repositories.wft_repository import WFTRepository
from src.services.backtest_write_service import BacktestWriteService
from src.services.bt_wft_records import compute_pf_from_records, dataframe_to_bt_records


@dataclass
class _SimpleMetrics:
    executed_trades: int
    total_profit_r: float
    win_rate_pct: float
    max_total_dd_pct: float = 0.0


def _export_csv_enabled() -> bool:
    return os.environ.get("BT_WRITE_CSV", "0").strip().lower() in {"1", "true", "yes"}


def _metrics_from_records(records: list[dict[str, Any]]) -> _SimpleMetrics:
    executed = [r for r in records if r.get("trade_result") not in (None, "NOT_EXECUTED")]
    if not executed:
        return _SimpleMetrics(0, 0.0, 0.0)
    rs = [float(r.get("profit_r") or r.get("result_r") or 0.0) for r in executed]
    wins = sum(1 for r in rs if r > 0)
    return _SimpleMetrics(
        executed_trades=len(executed),
        total_profit_r=float(sum(rs)),
        win_rate_pct=wins / len(executed) * 100.0,
    )


def persist_dataframe_bt(
    df: pd.DataFrame,
    *,
    strategy: str,
    logical_path: str,
    export_csv_path: Path | None = None,
) -> str:
    records = dataframe_to_bt_records(df, default_strategy=strategy)
    if not records:
        return ""
    metrics = _metrics_from_records(records)
    service = BacktestWriteService()
    try:
        run_id = service.save_bt_result(
            records,
            metrics=metrics,
            strategy=strategy,
            output_path=logical_path,
            started_at=str(records[0].get("timestamp", ""))[:19] or None,
            finished_at=str(records[-1].get("timestamp", ""))[:19] or None,
        )
    finally:
        service.close()

    if export_csv_path is not None and _export_csv_enabled():
        export_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(export_csv_path, index=False, encoding="utf-8-sig")
    return run_id


def persist_standalone_wft(
    wft_df: pd.DataFrame,
    windows: list[Any],
    *,
    strategy: str,
    wft_id_prefix: str,
    is_months: int,
    oos_months: int,
    step_months: int,
    window_id_column: str = "wft_window_id",
    r_column: str = "result_r",
) -> str:
    wft_id = f"{wft_id_prefix}_{uuid.uuid4().hex[:10]}"
    repo = WFTRepository()
    try:
        repo.save_wft(
            {
                "wft_id": wft_id,
                "strategy": strategy,
                "is_months": is_months,
                "oos_months": oos_months,
                "step_months": step_months,
                "total_windows": len(windows),
                "created_at": utc_now_iso(),
            }
        )

        oos_rs: list[float] = []
        oos_pfs: list[float] = []
        oos_dds: list[float] = []
        pass_flags: list[int] = []
        all_trades: list[dict[str, Any]] = []

        for spec in windows:
            window_no = int(getattr(spec, "window_id", 0))
            window_key = f"{wft_id}_w{window_no:04d}"
            if window_id_column in wft_df.columns:
                seg = wft_df[wft_df[window_id_column] == window_no]
            else:
                seg = wft_df.iloc[0:0]

            records = dataframe_to_bt_records(seg, default_strategy=strategy, r_column=r_column)
            total_r = sum(float(r.get("profit_r") or 0.0) for r in records)
            pf_raw = compute_pf_from_records(records) if records else 0.0
            pf = None if pf_raw == float("inf") else pf_raw

            repo.save_window(
                {
                    "window_id": window_key,
                    "wft_id": wft_id,
                    "window_no": window_no,
                    "is_start": _ts(getattr(spec, "is_start", None)),
                    "is_end": _ts(getattr(spec, "is_end", None)),
                    "oos_start": _ts(getattr(spec, "oos_start", None)),
                    "oos_end": _ts(getattr(spec, "oos_end", None)),
                    "total_r": total_r,
                    "pf": pf,
                    "max_dd": None,
                    "pass_flag": 1 if total_r >= 0 else 0,
                }
            )

            for record in records:
                base = record.copy()
                trade_id = base.get("trade_id") or uuid.uuid4().hex
                all_trades.append(
                    {
                        "trade_id": f"{window_key}:{trade_id}",
                        "window_id": window_key,
                        "wft_id": wft_id,
                        "strategy": base.get("setup_type") or strategy,
                        "symbol": base.get("pair"),
                        "open_time": base.get("timestamp"),
                        "close_time": base.get("timestamp"),
                        "direction": base.get("direction"),
                        "r_multiple": base.get("profit_r"),
                        "pnl": None,
                        "exit_reason": base.get("trade_result"),
                    }
                )

            oos_rs.append(total_r)
            if pf is not None:
                oos_pfs.append(float(pf))
            pass_flags.append(1 if total_r >= 0 else 0)

        if all_trades:
            repo.save_trades(all_trades)

        repo.save_summary(
            {
                "wft_id": wft_id,
                "total_oos_r": sum(oos_rs),
                "mean_oos_pf": (sum(oos_pfs) / len(oos_pfs)) if oos_pfs else None,
                "mean_oos_dd": None,
                "pass_rate": (sum(pass_flags) / len(pass_flags) * 100.0) if pass_flags else None,
                "stability": {},
                "created_at": utc_now_iso(),
            }
        )
        return wft_id
    finally:
        repo.close()


def load_dataframe_from_sqlite(logical_path: str) -> pd.DataFrame:
    repo = BacktestRepository()
    try:
        row = repo._db.query(
            "SELECT run_id FROM bt_runs WHERE notes=? ORDER BY finished_at DESC LIMIT 1",
            (logical_path,),
            one=True,
        )
        if row is None:
            raise FileNotFoundError(f"No SQLite BT run for logical_path={logical_path}")
        trades = repo.get_trades(str(row["run_id"]))
        if not trades:
            raise FileNotFoundError(f"No trades in SQLite for logical_path={logical_path}")
        df = pd.DataFrame(trades)
        df = df.rename(
            columns={
                "symbol": "pair",
                "open_time": "timestamp",
                "r_multiple": "profit_r",
                "exit_reason": "trade_result",
                "strategy": "setup_type",
            }
        )
        df["result_r"] = df["profit_r"]
        return df
    finally:
        repo.close()


def _ts(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:19].replace("T", " ")
