"""AGE v3 — Predictive Governor schema."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

AGE_V3_SCHEMA_VERSION = "3.0"

AGE_V3_DDL = """
CREATE TABLE IF NOT EXISTS governor_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    forecast_horizon TEXT NOT NULL,
    health_forecast TEXT,
    risk_forecast TEXT,
    recovery_probability TEXT,
    future_state TEXT,
    confidence REAL,
    recommendation_json TEXT,
    profile_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_governor_forecasts_ts
    ON governor_forecasts(timestamp DESC);

CREATE TABLE IF NOT EXISTS governor_predictive_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    details_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_governor_predictive_alerts_ts
    ON governor_predictive_alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_governor_predictive_alerts_type
    ON governor_predictive_alerts(alert_type);
"""


def apply_age_v3_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "governor_forecasts"):
        conn.executescript(AGE_V3_DDL)
    _set_schema_meta(conn, "age_v3_schema_version", AGE_V3_SCHEMA_VERSION)
