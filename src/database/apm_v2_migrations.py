"""APM v2 schema — executive memory and decision outcomes."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

APM_V2_SCHEMA_VERSION = "2.0.0"

APM_V2_DDL = """
CREATE TABLE IF NOT EXISTS executive_decision_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    decision_type TEXT,
    predicted_benefit REAL,
    actual_benefit REAL,
    predicted_risk REAL,
    actual_risk REAL,
    success_score REAL,
    outcome_class TEXT,
    evaluation_date TEXT
);

CREATE INDEX IF NOT EXISTS idx_executive_decision_outcomes_profile
    ON executive_decision_outcomes(profile_id);

CREATE TABLE IF NOT EXISTS executive_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    category TEXT,
    title TEXT,
    summary TEXT,
    success_rate REAL,
    confidence REAL,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_executive_memory_profile
    ON executive_memory(profile_id);

CREATE TABLE IF NOT EXISTS executive_lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    source_module TEXT,
    lesson_type TEXT,
    description TEXT,
    impact_score REAL,
    confidence REAL,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_executive_lessons_profile
    ON executive_lessons(profile_id);
"""


def apply_apm_v2_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "executive_decision_outcomes"):
        conn.executescript(APM_V2_DDL)
    _set_schema_meta(conn, "apm_v2_schema_version", APM_V2_SCHEMA_VERSION)
