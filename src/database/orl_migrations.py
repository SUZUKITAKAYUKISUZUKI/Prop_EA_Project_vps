"""ORL v1 schema — readiness, audit, health history."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

ORL_SCHEMA_VERSION = "1.0.0"

ORL_DDL = """
CREATE TABLE IF NOT EXISTS operational_readiness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    readiness_score REAL,
    readiness_status TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_operational_readiness_profile
    ON operational_readiness(profile_id);

CREATE TABLE IF NOT EXISTS operational_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    category TEXT,
    severity TEXT,
    message TEXT
);

CREATE INDEX IF NOT EXISTS idx_operational_audit_log_profile
    ON operational_audit_log(profile_id);

CREATE TABLE IF NOT EXISTS system_health_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    system_health REAL,
    readiness_score REAL
);

CREATE INDEX IF NOT EXISTS idx_system_health_history_profile
    ON system_health_history(profile_id);
"""


def apply_orl_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "operational_readiness"):
        conn.executescript(ORL_DDL)
    _set_schema_meta(conn, "orl_schema_version", ORL_SCHEMA_VERSION)
