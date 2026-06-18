"""AI Governor Engine — SQLite schema."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _add_column, _column_exists, _set_schema_meta, _table_exists

AGE_SCHEMA_VERSION = "2.0"

AGE_DDL = """
CREATE TABLE IF NOT EXISTS governor_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    decision TEXT,
    confidence REAL NOT NULL,
    reason_json TEXT,
    profile TEXT,
    state TEXT,
    profile_id TEXT,
    source_state TEXT,
    executed INTEGER DEFAULT 0,
    created_by TEXT DEFAULT 'age_engine'
);

CREATE INDEX IF NOT EXISTS idx_governor_decisions_ts
    ON governor_decisions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_governor_decisions_profile
    ON governor_decisions(profile);
CREATE INDEX IF NOT EXISTS idx_governor_decisions_type
    ON governor_decisions(decision_type);

CREATE TABLE IF NOT EXISTS governor_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT,
    category TEXT,
    priority TEXT NOT NULL,
    recommendation TEXT,
    confidence REAL NOT NULL,
    expected_benefit REAL,
    expected_risk REAL,
    reason_json TEXT,
    status TEXT DEFAULT 'OPEN'
);

CREATE INDEX IF NOT EXISTS idx_governor_recommendations_status
    ON governor_recommendations(status);
CREATE INDEX IF NOT EXISTS idx_governor_recommendations_ts
    ON governor_recommendations(timestamp DESC);

CREATE TABLE IF NOT EXISTS governor_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    health_score REAL NOT NULL,
    health_status TEXT NOT NULL,
    state TEXT NOT NULL,
    profile TEXT,
    profile_id TEXT,
    risk_level TEXT,
    risk_score REAL
);

CREATE INDEX IF NOT EXISTS idx_governor_health_ts
    ON governor_health(timestamp DESC);
"""


def apply_age_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "governor_decisions"):
        conn.executescript(AGE_DDL)
    else:
        _upgrade_age_schema(conn)
    _set_schema_meta(conn, "age_schema_version", AGE_SCHEMA_VERSION)


def _upgrade_age_schema(conn: sqlite3.Connection) -> None:
    decision_cols = (
        "decision TEXT",
        "profile TEXT",
        "state TEXT",
        "executed INTEGER DEFAULT 0",
        "created_by TEXT DEFAULT 'age_engine'",
    )
    for col in decision_cols:
        _add_column(conn, "governor_decisions", col)

    recommendation_cols = (
        "action TEXT",
        "expected_benefit REAL",
        "expected_risk REAL",
        "reason_json TEXT",
    )
    for col in recommendation_cols:
        _add_column(conn, "governor_recommendations", col)

    health_cols = (
        "profile TEXT",
        "risk_score REAL",
    )
    for col in health_cols:
        _add_column(conn, "governor_health", col)

    if _column_exists(conn, "governor_decisions", "profile_id") and _column_exists(conn, "governor_decisions", "profile"):
        conn.execute(
            """
            UPDATE governor_decisions
            SET profile = profile_id
            WHERE profile IS NULL AND profile_id IS NOT NULL
            """
        )
    if _column_exists(conn, "governor_decisions", "source_state") and _column_exists(conn, "governor_decisions", "state"):
        conn.execute(
            """
            UPDATE governor_decisions
            SET state = source_state
            WHERE state IS NULL AND source_state IS NOT NULL
            """
        )
    if _column_exists(conn, "governor_health", "profile_id") and _column_exists(conn, "governor_health", "profile"):
        conn.execute(
            """
            UPDATE governor_health
            SET profile = profile_id
            WHERE profile IS NULL AND profile_id IS NOT NULL
            """
        )
