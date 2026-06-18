"""CACE v1.5 schema upgrades."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import (
    _add_column,
    _column_exists,
    _set_schema_meta,
    _table_exists,
)

CACE_V15_SCHEMA_VERSION = "1.5"

CONFIDENCE_REGIME_DDL = """
CREATE TABLE IF NOT EXISTS confidence_regime_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT,
    regime TEXT,
    confidence_modifier REAL,
    metrics_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_confidence_regime_snapshots_profile
    ON confidence_regime_snapshots(profile_id);
CREATE INDEX IF NOT EXISTS idx_confidence_regime_snapshots_ts
    ON confidence_regime_snapshots(timestamp DESC);
"""


def apply_cace_v15_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "allocation_confidence"):
        from src.database.cace_migrations import apply_cace_migrations

        apply_cace_migrations(conn)

    _upgrade_allocation_confidence(conn)
    _upgrade_strategy_confidence(conn)
    _upgrade_confidence_history(conn)

    if not _table_exists(conn, "confidence_regime_snapshots"):
        conn.executescript(CONFIDENCE_REGIME_DDL)

    _set_schema_meta(conn, "cace_v15_schema_version", CACE_V15_SCHEMA_VERSION)


def _upgrade_allocation_confidence(conn: sqlite3.Connection) -> None:
    columns = (
        "historical_reliability REAL",
        "mc_stability REAL",
        "forecast_stability REAL",
        "portfolio_health REAL",
        "lifecycle_quality REAL",
        "breakdown_json TEXT",
    )
    for ddl in columns:
        _add_column(conn, "allocation_confidence", ddl)


def _upgrade_strategy_confidence(conn: sqlite3.Connection) -> None:
    _add_column(conn, "strategy_confidence", "breakdown_json TEXT")


def _upgrade_confidence_history(conn: sqlite3.Connection) -> None:
    for ddl in ("trend TEXT", "trend_strength REAL"):
        _add_column(conn, "confidence_history", ddl)
