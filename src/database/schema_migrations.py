"""Incremental portfolio_os.db schema upgrades."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src.database.data_source import (
    FEATURE_LOG_SCHEMA_VERSION,
    PORTFOLIO_DB_SCHEMA_VERSION,
    infer_source_from_path,
    infer_source_from_run_type,
)

SCHEMA_META_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _add_column(conn: sqlite3.Connection, table: str, ddl: str) -> None:
    if not _table_exists(conn, table):
        return
    column = ddl.split()[0]
    if _column_exists(conn, table, column):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _set_schema_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (key, value, _utc_now_iso()),
    )


def _backfill_run_sources(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "runs"):
        return
    rows = conn.execute(
        "SELECT run_id, run_type, description, source FROM runs"
    ).fetchall()
    for row in rows:
        run_id = int(row[0])
        run_type = row[1]
        description = row[2]
        current = row[3]
        if current and current not in ("", "BACKTEST"):
            continue
        inferred = infer_source_from_run_type(run_type, description)
        if description:
            path_inferred = infer_source_from_path(description, csv_kind=run_type)
            if path_inferred != "BACKTEST" or inferred == "BACKTEST":
                inferred = path_inferred
        conn.execute("UPDATE runs SET source=? WHERE run_id=?", (inferred, run_id))


def _propagate_sources(conn: sqlite3.Connection) -> None:
    if _table_exists(conn, "trades") and _column_exists(conn, "trades", "source"):
        conn.execute(
            """
            UPDATE trades
            SET source = (
                SELECT runs.source FROM runs WHERE runs.run_id = trades.run_id
            )
            WHERE source IS NULL OR source = 'BACKTEST'
            """
        )
    if _table_exists(conn, "features") and _column_exists(conn, "features", "source"):
        conn.execute(
            """
            UPDATE features
            SET source = (
                SELECT runs.source FROM runs WHERE runs.run_id = features.run_id
            )
            WHERE source IS NULL OR source = 'BACKTEST'
            """
        )


def apply_portfolio_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_META_DDL)

    _add_column(conn, "runs", "source TEXT NOT NULL DEFAULT 'BACKTEST'")
    _add_column(conn, "runs", f"schema_version INTEGER NOT NULL DEFAULT {PORTFOLIO_DB_SCHEMA_VERSION}")

    _add_column(conn, "trades", "source TEXT NOT NULL DEFAULT 'BACKTEST'")
    _add_column(conn, "features", "source TEXT NOT NULL DEFAULT 'BACKTEST'")
    _add_column(conn, "features", f"schema_version INTEGER NOT NULL DEFAULT {FEATURE_LOG_SCHEMA_VERSION}")

    _add_column(conn, "trade_events", "source TEXT NOT NULL DEFAULT 'LIVE'")
    _add_column(conn, "trade_events", f"schema_version INTEGER NOT NULL DEFAULT {FEATURE_LOG_SCHEMA_VERSION}")

    _backfill_run_sources(conn)
    _propagate_sources(conn)

    if _table_exists(conn, "trade_events"):
        conn.execute(
            "UPDATE trade_events SET source='LIVE' WHERE source IS NULL OR source=''"
        )

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_source ON runs(source);
        CREATE INDEX IF NOT EXISTS idx_trades_source ON trades(source);
        CREATE INDEX IF NOT EXISTS idx_features_source ON features(source);
        CREATE INDEX IF NOT EXISTS idx_trade_events_source ON trade_events(source);
        """
    )

    _set_schema_meta(conn, "portfolio_db_schema_version", str(PORTFOLIO_DB_SCHEMA_VERSION))
    _set_schema_meta(conn, "feature_log_schema_version", str(FEATURE_LOG_SCHEMA_VERSION))
    conn.commit()
