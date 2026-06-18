"""Native SQLite persistence for walk-forward test runs."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class WFTRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def save_wft(self, wft: dict[str, Any]) -> str:
        wft_id = str(wft["wft_id"])
        self._db.portfolio.execute(
            """
            INSERT INTO wft_runs (
                wft_id, strategy, is_months, oos_months, step_months, total_windows, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wft_id) DO UPDATE SET
                strategy=excluded.strategy,
                is_months=excluded.is_months,
                oos_months=excluded.oos_months,
                step_months=excluded.step_months,
                total_windows=excluded.total_windows,
                created_at=excluded.created_at
            """,
            (
                wft_id,
                wft.get("strategy"),
                int(wft.get("is_months") or 0),
                int(wft.get("oos_months") or 0),
                int(wft.get("step_months") or 0),
                int(wft.get("total_windows") or 0),
                wft.get("created_at") or utc_now_iso(),
            ),
        )
        self._db.portfolio.commit()
        return wft_id

    def save_window(self, window: dict[str, Any]) -> str:
        window_id = str(window["window_id"])
        self._db.portfolio.execute(
            """
            INSERT INTO wft_windows (
                window_id, wft_id, window_no, is_start, is_end, oos_start, oos_end,
                total_r, pf, max_dd, pass_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(window_id) DO UPDATE SET
                wft_id=excluded.wft_id,
                window_no=excluded.window_no,
                is_start=excluded.is_start,
                is_end=excluded.is_end,
                oos_start=excluded.oos_start,
                oos_end=excluded.oos_end,
                total_r=excluded.total_r,
                pf=excluded.pf,
                max_dd=excluded.max_dd,
                pass_flag=excluded.pass_flag
            """,
            (
                window_id,
                window["wft_id"],
                int(window.get("window_no") or 0),
                window.get("is_start"),
                window.get("is_end"),
                window.get("oos_start"),
                window.get("oos_end"),
                window.get("total_r"),
                window.get("pf"),
                window.get("max_dd"),
                window.get("pass_flag"),
            ),
        )
        self._db.portfolio.commit()
        return window_id

    def save_trades(self, trades: list[dict[str, Any]]) -> int:
        count = 0
        for trade in trades:
            self._db.portfolio.execute(
                """
                INSERT INTO wft_trades (
                    trade_id, window_id, wft_id, strategy, symbol, open_time, close_time,
                    direction, r_multiple, pnl, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    window_id=excluded.window_id,
                    wft_id=excluded.wft_id,
                    strategy=excluded.strategy,
                    symbol=excluded.symbol,
                    open_time=excluded.open_time,
                    close_time=excluded.close_time,
                    direction=excluded.direction,
                    r_multiple=excluded.r_multiple,
                    pnl=excluded.pnl,
                    exit_reason=excluded.exit_reason
                """,
                (
                    trade["trade_id"],
                    trade["window_id"],
                    trade["wft_id"],
                    trade.get("strategy"),
                    trade.get("symbol"),
                    trade.get("open_time"),
                    trade.get("close_time"),
                    trade.get("direction"),
                    trade.get("r_multiple"),
                    trade.get("pnl"),
                    trade.get("exit_reason"),
                ),
            )
            count += 1
        self._db.portfolio.commit()
        return count

    def save_summary(self, summary: dict[str, Any]) -> str:
        wft_id = str(summary["wft_id"])
        stability = summary.get("stability")
        payload = json.dumps(stability, ensure_ascii=False) if isinstance(stability, dict) else stability
        self._db.portfolio.execute(
            """
            INSERT INTO wft_summary (
                wft_id, total_oos_r, mean_oos_pf, mean_oos_dd, pass_rate, stability_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wft_id) DO UPDATE SET
                total_oos_r=excluded.total_oos_r,
                mean_oos_pf=excluded.mean_oos_pf,
                mean_oos_dd=excluded.mean_oos_dd,
                pass_rate=excluded.pass_rate,
                stability_json=excluded.stability_json,
                created_at=excluded.created_at
            """,
            (
                wft_id,
                summary.get("total_oos_r"),
                summary.get("mean_oos_pf"),
                summary.get("mean_oos_dd"),
                summary.get("pass_rate"),
                payload,
                summary.get("created_at") or utc_now_iso(),
            ),
        )
        self._db.portfolio.commit()
        return wft_id

    def load_wft(self, wft_id: str) -> dict[str, Any] | None:
        row = self._db.query("SELECT * FROM wft_runs WHERE wft_id=?", (wft_id,), one=True)
        return dict(row) if row else None

    def load_windows(self, wft_id: str) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM wft_windows WHERE wft_id=? ORDER BY window_no ASC",
            (wft_id,),
        )
        return [dict(r) for r in rows]

    def load_summary(self, wft_id: str) -> dict[str, Any] | None:
        row = self._db.query("SELECT * FROM wft_summary WHERE wft_id=?", (wft_id,), one=True)
        return dict(row) if row else None

    def count_windows(self, wft_id: str | None = None) -> int:
        if wft_id:
            row = self._db.query("SELECT COUNT(*) AS c FROM wft_windows WHERE wft_id=?", (wft_id,), one=True)
        else:
            row = self._db.query("SELECT COUNT(*) AS c FROM wft_windows", one=True)
        return int(row["c"]) if row else 0

    def load_latest_for_strategy(self, strategy: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT r.*, s.mean_oos_pf, s.pass_rate, s.stability_json
            FROM wft_runs r
            LEFT JOIN wft_summary s ON s.wft_id = r.wft_id
            WHERE r.strategy = ?
            ORDER BY r.created_at DESC
            LIMIT 1
            """,
            (strategy,),
            one=True,
        )
        return dict(row) if row else None
