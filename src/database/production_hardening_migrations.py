"""Portfolio OS RC1 schema — production hardening tables."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

PRODUCTION_SCHEMA_VERSION = "1.0.0"

PRODUCTION_DDL = """
CREATE TABLE IF NOT EXISTS production_readiness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    production_readiness REAL,
    resilience_score REAL,
    production_status TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_production_readiness_profile
    ON production_readiness(profile_id);

CREATE TABLE IF NOT EXISTS production_validation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    end_to_end_score REAL,
    chain_health REAL,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_production_validation_history_profile
    ON production_validation_history(profile_id);

CREATE TABLE IF NOT EXISTS production_benchmark_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    benchmark_score REAL,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_production_benchmark_history_profile
    ON production_benchmark_history(profile_id);

CREATE TABLE IF NOT EXISTS production_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    scenario TEXT,
    recovered INTEGER DEFAULT 0,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_production_failures_profile
    ON production_failures(profile_id);

CREATE TABLE IF NOT EXISTS production_resilience_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    resilience_score REAL,
    failure_recovery REAL
);

CREATE INDEX IF NOT EXISTS idx_production_resilience_history_profile
    ON production_resilience_history(profile_id);
"""


def apply_production_hardening_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "production_readiness"):
        conn.executescript(PRODUCTION_DDL)
    _set_schema_meta(conn, "production_hardening_schema_version", PRODUCTION_SCHEMA_VERSION)
