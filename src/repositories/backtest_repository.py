"""Native SQLite persistence for backtest runs and trades."""
from __future__ import annotations

from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class BacktestRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def save_run(self, run: dict[str, Any]) -> str:
        run_id = str(run["run_id"])
        self._db.portfolio.execute(
            """
            INSERT INTO bt_runs (
                run_id, strategy, symbol, started_at, finished_at,
                total_trades, total_r, pf, win_rate, avg_r, max_dd, sharpe,
                source_version, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                strategy=excluded.strategy,
                symbol=excluded.symbol,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                total_trades=excluded.total_trades,
                total_r=excluded.total_r,
                pf=excluded.pf,
                win_rate=excluded.win_rate,
                avg_r=excluded.avg_r,
                max_dd=excluded.max_dd,
                sharpe=excluded.sharpe,
                source_version=excluded.source_version,
                notes=excluded.notes
            """,
            (
                run_id,
                run.get("strategy"),
                run.get("symbol"),
                run.get("started_at"),
                run.get("finished_at"),
                int(run.get("total_trades") or 0),
                run.get("total_r"),
                run.get("pf"),
                run.get("win_rate"),
                run.get("avg_r"),
                run.get("max_dd"),
                run.get("sharpe"),
                run.get("source_version"),
                run.get("notes"),
            ),
        )
        self._db.portfolio.commit()
        return run_id

    def save_trades(self, run_id: str, trades: list[dict[str, Any]]) -> int:
        count = 0
        for trade in trades:
            self._db.portfolio.execute(
                """
                INSERT INTO bt_trades (
                    trade_id, run_id, strategy, symbol, open_time, close_time,
                    direction, r_multiple, pnl, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    run_id=excluded.run_id,
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
                    run_id,
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

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._db.query("SELECT * FROM bt_runs WHERE run_id=?", (run_id,), one=True)
        return dict(row) if row else None

    def get_trades(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM bt_trades WHERE run_id=? ORDER BY open_time ASC, trade_id ASC",
            (run_id,),
        )
        return [dict(r) for r in rows]

    def count_runs(self) -> int:
        row = self._db.query("SELECT COUNT(*) AS c FROM bt_runs", one=True)
        return int(row["c"]) if row else 0

    def register_legacy_run(
        self,
        *,
        run_id: str,
        strategy: str | None,
        description: str,
        legacy_run_id: int,
    ) -> None:
        self._db.portfolio.execute(
            """
            INSERT INTO bt_run_legacy_map (bt_run_id, legacy_run_id, description, linked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(bt_run_id) DO UPDATE SET
                legacy_run_id=excluded.legacy_run_id,
                description=excluded.description,
                linked_at=excluded.linked_at
            """,
            (run_id, legacy_run_id, description, utc_now_iso()),
        )
        self._db.portfolio.commit()

    def get_legacy_run_id(self, bt_run_id: str) -> int | None:
        row = self._db.query(
            "SELECT legacy_run_id FROM bt_run_legacy_map WHERE bt_run_id=?",
            (bt_run_id,),
            one=True,
        )
        return int(row["legacy_run_id"]) if row else None
