"""APM v1 schema — execution queue and executive reports."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

APM_SCHEMA_VERSION = "1.0.0"

APM_DDL = """
CREATE TABLE IF NOT EXISTS apm_execution_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    action_type TEXT,
    strategy TEXT,
    confidence REAL,
    expected_benefit_pct REAL,
    expected_risk_pct REAL,
    status TEXT,
    rejection_reason TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_apm_execution_queue_profile
    ON apm_execution_queue(profile_id);
CREATE INDEX IF NOT EXISTS idx_apm_execution_queue_status
    ON apm_execution_queue(status);

CREATE TABLE IF NOT EXISTS apm_executive_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    executive_score REAL,
    executive_category TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_apm_executive_reports_profile
    ON apm_executive_reports(profile_id);

CREATE TABLE IF NOT EXISTS apm_roadmaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    horizon TEXT,
    action_type TEXT,
    strategy TEXT,
    description TEXT,
    confidence REAL,
    status TEXT
);

CREATE INDEX IF NOT EXISTS idx_apm_roadmaps_profile
    ON apm_roadmaps(profile_id);

CREATE TABLE IF NOT EXISTS apm_opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    strategy TEXT,
    portfolio_fit REAL,
    lifecycle_score REAL,
    current_allocation_pct REAL,
    recommended_allocation_pct REAL,
    message TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_apm_opportunities_profile
    ON apm_opportunities(profile_id);

CREATE TABLE IF NOT EXISTS apm_risk_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    strategy TEXT,
    risk_score REAL,
    dd_contribution_pct REAL,
    health_impact REAL,
    message TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_apm_risk_alerts_profile
    ON apm_risk_alerts(profile_id);
"""


def apply_apm_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "apm_execution_queue"):
        conn.executescript(APM_DDL)
    _set_schema_meta(conn, "apm_schema_version", APM_SCHEMA_VERSION)
