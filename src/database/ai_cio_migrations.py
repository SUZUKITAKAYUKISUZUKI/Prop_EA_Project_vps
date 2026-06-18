"""AI CIO v1 schema — reports, recommendations, opinions."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

AI_CIO_SCHEMA_VERSION = "1.0.0"

AI_CIO_DDL = """
CREATE TABLE IF NOT EXISTS cio_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    cio_score REAL,
    cio_opinion TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_cio_reports_profile
    ON cio_reports(profile_id);

CREATE TABLE IF NOT EXISTS cio_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    category TEXT,
    priority INTEGER,
    action TEXT,
    description TEXT,
    confidence REAL,
    requires_approval INTEGER DEFAULT 1,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_cio_recommendations_profile
    ON cio_recommendations(profile_id);

CREATE TABLE IF NOT EXISTS cio_opinions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    cio_opinion TEXT,
    cio_score REAL
);

CREATE INDEX IF NOT EXISTS idx_cio_opinions_profile
    ON cio_opinions(profile_id);
"""


def apply_ai_cio_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "cio_reports"):
        conn.executescript(AI_CIO_DDL)
    _set_schema_meta(conn, "ai_cio_schema_version", AI_CIO_SCHEMA_VERSION)
