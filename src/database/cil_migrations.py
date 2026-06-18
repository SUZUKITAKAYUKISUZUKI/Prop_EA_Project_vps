"""CIL v1 schema — intelligence snapshots and score history."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

CIL_SCHEMA_VERSION = "1.0.0"

CIL_DDL = """
CREATE TABLE IF NOT EXISTS cio_intelligence_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    executive_score REAL,
    investment_state_json TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_cio_intelligence_snapshots_profile
    ON cio_intelligence_snapshots(profile_id);

CREATE TABLE IF NOT EXISTS executive_investment_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    executive_score REAL,
    opportunity_score REAL,
    risk_score REAL,
    confidence_score REAL,
    capital_efficiency REAL
);

CREATE INDEX IF NOT EXISTS idx_executive_investment_scores_profile
    ON executive_investment_scores(profile_id);
"""


def apply_cil_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "cio_intelligence_snapshots"):
        conn.executescript(CIL_DDL)
    _set_schema_meta(conn, "cil_schema_version", CIL_SCHEMA_VERSION)
