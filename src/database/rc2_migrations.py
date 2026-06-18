"""RC2 Live Operations schema."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

RC2_SCHEMA_VERSION = "1.0.0"

RC2_DDL = """
CREATE TABLE IF NOT EXISTS daily_briefings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    briefing_date TEXT,
    operational_score REAL,
    live_readiness REAL,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_daily_briefings_profile
    ON daily_briefings(profile_id);

CREATE TABLE IF NOT EXISTS daily_digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    digest_date TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_daily_digests_profile
    ON daily_digests(profile_id);

CREATE TABLE IF NOT EXISTS operational_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    level TEXT,
    message TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_operational_alerts_profile
    ON operational_alerts(profile_id);

CREATE TABLE IF NOT EXISTS anomaly_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    anomaly_type TEXT,
    severity TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_anomaly_history_profile
    ON anomaly_history(profile_id);

CREATE TABLE IF NOT EXISTS live_operations_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    operational_score REAL,
    live_readiness REAL,
    user_action_load INTEGER,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_operations_history_profile
    ON live_operations_history(profile_id);
"""


def apply_rc2_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "daily_briefings"):
        conn.executescript(RC2_DDL)
    _set_schema_meta(conn, "rc2_schema_version", RC2_SCHEMA_VERSION)
