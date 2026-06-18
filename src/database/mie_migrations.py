"""MIE v1 schema — module trust, drift, rankings."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

MIE_SCHEMA_VERSION = "1.0.0"

MIE_DDL = """
CREATE TABLE IF NOT EXISTS module_trust_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    module TEXT NOT NULL,
    trust_score REAL,
    category TEXT
);

CREATE INDEX IF NOT EXISTS idx_module_trust_history_profile
    ON module_trust_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_module_trust_history_module
    ON module_trust_history(module);

CREATE TABLE IF NOT EXISTS module_drift_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    module TEXT NOT NULL,
    previous_score REAL,
    current_score REAL,
    delta REAL,
    alert_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_module_drift_alerts_profile
    ON module_drift_alerts(profile_id);

CREATE TABLE IF NOT EXISTS module_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    module TEXT NOT NULL,
    rank INTEGER,
    trust_score REAL
);

CREATE INDEX IF NOT EXISTS idx_module_rankings_profile
    ON module_rankings(profile_id);

CREATE TABLE IF NOT EXISTS mie_self_improvement_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    module TEXT,
    issue TEXT,
    recommendation TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_mie_self_improvement_profile
    ON mie_self_improvement_notes(profile_id);
"""


def apply_mie_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "module_trust_history"):
        conn.executescript(MIE_DDL)
    _set_schema_meta(conn, "mie_schema_version", MIE_SCHEMA_VERSION)
