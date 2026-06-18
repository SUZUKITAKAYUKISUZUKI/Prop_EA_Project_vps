"""CACE — Capital Allocation Confidence Engine schema."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

CACE_SCHEMA_VERSION = "1.0"

CACE_DDL = """
CREATE TABLE IF NOT EXISTS allocation_confidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT,
    allocation_json TEXT,
    confidence REAL,
    category TEXT,
    expected_r REAL,
    expected_pf REAL,
    expected_dd REAL,
    reason_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_allocation_confidence_ts
    ON allocation_confidence(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_allocation_confidence_profile
    ON allocation_confidence(profile_id);

CREATE TABLE IF NOT EXISTS strategy_confidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    confidence REAL,
    portfolio_fit REAL,
    lifecycle_stage TEXT,
    reason_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_strategy_confidence_ts
    ON strategy_confidence(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_confidence_strategy
    ON strategy_confidence(strategy);

CREATE TABLE IF NOT EXISTS confidence_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT,
    confidence REAL,
    category TEXT,
    snapshot_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_confidence_history_ts
    ON confidence_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_confidence_history_profile
    ON confidence_history(profile_id);
"""


def apply_cace_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "allocation_confidence"):
        conn.executescript(CACE_DDL)
    _set_schema_meta(conn, "cace_schema_version", CACE_SCHEMA_VERSION)
