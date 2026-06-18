"""Portfolio Risk Attribution Engine — SQLite schema."""
from __future__ import annotations

import sqlite3

RISK_ATTRIBUTION_SCHEMA_VERSION = "1.0"

RISK_ATTRIBUTION_DDL = """
CREATE TABLE IF NOT EXISTS risk_attribution_reports (
    report_id TEXT PRIMARY KEY,
    source_run_id TEXT,
    profile_id TEXT,
    generated_at TEXT,
    total_r REAL,
    total_dd REAL,
    pf REAL,
    win_rate REAL,
    report_json TEXT
);

CREATE TABLE IF NOT EXISTS risk_attribution_cache (
    cache_key TEXT PRIMARY KEY,
    cache_value TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_risk_attr_reports_profile
    ON risk_attribution_reports(profile_id);
CREATE INDEX IF NOT EXISTS idx_risk_attr_reports_generated
    ON risk_attribution_reports(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_attr_reports_run
    ON risk_attribution_reports(source_run_id);
"""


def apply_risk_attribution_migrations(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_attribution_reports'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript(RISK_ATTRIBUTION_DDL)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        ("risk_attribution_schema_version", RISK_ATTRIBUTION_SCHEMA_VERSION),
    )
