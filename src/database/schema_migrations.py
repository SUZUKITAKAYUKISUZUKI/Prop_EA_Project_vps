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
from src.database.profile_migrations import (
    PROFILE_MANAGER_SCHEMA_VERSION,
    apply_profile_manager_migrations,
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


def _get_schema_meta(conn: sqlite3.Connection, key: str) -> str | None:
    if not _table_exists(conn, "schema_meta"):
        return None
    row = conn.execute("SELECT value FROM schema_meta WHERE key=?", (key,)).fetchone()
    return str(row[0]) if row else None


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


def apply_portfolio_migrations(
    conn: sqlite3.Connection,
    *,
    run_legacy_backfill: bool = False,
) -> None:
    conn.executescript(SCHEMA_META_DDL)

    portfolio_ver = _get_schema_meta(conn, "portfolio_db_schema_version")
    profile_ver = _get_schema_meta(conn, "profile_manager_schema_version")
    schema_current = (
        not run_legacy_backfill
        and portfolio_ver == str(PORTFOLIO_DB_SCHEMA_VERSION)
        and profile_ver == PROFILE_MANAGER_SCHEMA_VERSION
    )

    if not schema_current:
        _add_column(conn, "runs", "source TEXT NOT NULL DEFAULT 'BACKTEST'")
        _add_column(conn, "runs", f"schema_version INTEGER NOT NULL DEFAULT {PORTFOLIO_DB_SCHEMA_VERSION}")

        _add_column(conn, "trades", "source TEXT NOT NULL DEFAULT 'BACKTEST'")
        _add_column(conn, "features", "source TEXT NOT NULL DEFAULT 'BACKTEST'")
        _add_column(conn, "features", f"schema_version INTEGER NOT NULL DEFAULT {FEATURE_LOG_SCHEMA_VERSION}")

        _add_column(conn, "trade_events", "source TEXT NOT NULL DEFAULT 'LIVE'")
        _add_column(conn, "trade_events", f"schema_version INTEGER NOT NULL DEFAULT {FEATURE_LOG_SCHEMA_VERSION}")

        if run_legacy_backfill:
            run_legacy_source_backfill(conn)

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
        apply_profile_manager_migrations(conn)

    from src.database.risk_attribution_migrations import apply_risk_attribution_migrations

    apply_risk_attribution_migrations(conn)
    from src.database.auto_switch_migrations import apply_auto_switch_migrations

    apply_auto_switch_migrations(conn)
    from src.database.state_analytics_migrations import apply_state_analytics_migrations

    apply_state_analytics_migrations(conn)
    from src.database.risk_attribution_v2_migrations import apply_risk_attribution_v2_migrations

    apply_risk_attribution_v2_migrations(conn)
    from src.database.paae_migrations import apply_paae_migrations

    apply_paae_migrations(conn)
    from src.database.pdts_migrations import apply_pdts_migrations

    apply_pdts_migrations(conn)
    from src.database.slm_migrations import apply_slm_migrations

    apply_slm_migrations(conn)
    from src.database.slm_v2_migrations import apply_slm_v2_migrations

    apply_slm_v2_migrations(conn)
    from src.database.slm_v3_migrations import apply_slm_v3_migrations

    apply_slm_v3_migrations(conn)
    from src.database.age_migrations import apply_age_migrations

    apply_age_migrations(conn)
    from src.database.age_v3_migrations import apply_age_v3_migrations

    apply_age_v3_migrations(conn)
    from src.database.age_v4_migrations import apply_age_v4_migrations

    apply_age_v4_migrations(conn)
    from src.database.cace_migrations import apply_cace_migrations

    apply_cace_migrations(conn)
    from src.database.cace_v15_migrations import apply_cace_v15_migrations

    apply_cace_v15_migrations(conn)
    from src.database.cace_v16_migrations import apply_cace_v16_migrations

    apply_cace_v16_migrations(conn)
    from src.database.cace_v17_migrations import apply_cace_v17_migrations

    apply_cace_v17_migrations(conn)
    from src.database.mie_migrations import apply_mie_migrations

    apply_mie_migrations(conn)
    from src.database.apm_migrations import apply_apm_migrations

    apply_apm_migrations(conn)
    from src.database.apm_v2_migrations import apply_apm_v2_migrations

    apply_apm_v2_migrations(conn)
    from src.database.cil_migrations import apply_cil_migrations

    apply_cil_migrations(conn)
    from src.database.ai_cio_migrations import apply_ai_cio_migrations

    apply_ai_cio_migrations(conn)
    from src.database.orl_migrations import apply_orl_migrations

    apply_orl_migrations(conn)
    from src.database.production_hardening_migrations import apply_production_hardening_migrations

    apply_production_hardening_migrations(conn)
    from src.database.rc2_migrations import apply_rc2_migrations

    apply_rc2_migrations(conn)
    conn.commit()


def run_legacy_source_backfill(conn: sqlite3.Connection) -> None:
    """One-time/heavy lineage backfill — run from tools/migrate_all.py only."""
    if _get_schema_meta(conn, "run_sources_backfill_v1") == "done":
        return
    _backfill_run_sources(conn)
    _propagate_sources(conn)
    _set_schema_meta(conn, "run_sources_backfill_v1", "done")
    if _get_schema_meta(conn, "trade_events_source_v1") != "done":
        if _table_exists(conn, "trade_events"):
            conn.execute(
                "UPDATE trade_events SET source='LIVE' WHERE source IS NULL OR source=''"
            )
        _set_schema_meta(conn, "trade_events_source_v1", "done")
    conn.commit()
