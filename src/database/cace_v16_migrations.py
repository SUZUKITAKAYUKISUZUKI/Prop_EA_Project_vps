"""CACE v1.6 schema — decay and consensus history."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _add_column, _column_exists, _set_schema_meta, _table_exists

CACE_V16_SCHEMA_VERSION = "1.6.1"

CACE_V16_DDL = """
CREATE TABLE IF NOT EXISTS confidence_decay_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    durability_score REAL,
    durability_category TEXT,
    half_life INTEGER,
    forecast_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_confidence_decay_history_profile
    ON confidence_decay_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_confidence_decay_history_ts
    ON confidence_decay_history(timestamp DESC);

CREATE TABLE IF NOT EXISTS confidence_consensus_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    recommended_action TEXT,
    consensus_score REAL,
    consensus_category TEXT,
    agreement_ratio REAL,
    agree_count INTEGER,
    total_modules INTEGER
);

CREATE INDEX IF NOT EXISTS idx_confidence_consensus_history_profile
    ON confidence_consensus_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_confidence_consensus_history_ts
    ON confidence_consensus_history(timestamp DESC);
"""


def apply_cace_v16_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "confidence_decay_history"):
        conn.executescript(CACE_V16_DDL)
    else:
        _upgrade_decay_table(conn)
        _upgrade_consensus_table(conn)
    _set_schema_meta(conn, "cace_v16_schema_version", CACE_V16_SCHEMA_VERSION)


def _upgrade_decay_table(conn: sqlite3.Connection) -> None:
    for ddl in (
        "durability_score REAL",
        "durability_category TEXT",
        "half_life INTEGER",
        "forecast_json TEXT",
    ):
        _add_column(conn, "confidence_decay_history", ddl)


def _upgrade_consensus_table(conn: sqlite3.Connection) -> None:
    for ddl in (
        "recommended_action TEXT",
        "consensus_category TEXT",
        "agreement_ratio REAL",
        "agree_count INTEGER",
        "total_modules INTEGER",
    ):
        _add_column(conn, "confidence_consensus_history", ddl)
    if _column_exists(conn, "confidence_consensus_history", "agreement_level") and not _column_exists(
        conn, "confidence_consensus_history", "consensus_category"
    ):
        conn.execute(
            """
            UPDATE confidence_consensus_history
            SET consensus_category = agreement_level
            WHERE consensus_category IS NULL AND agreement_level IS NOT NULL
            """
        )
