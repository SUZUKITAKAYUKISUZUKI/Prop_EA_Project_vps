"""Phase 3.0 Profile Manager — SQLite schema and default profile seed."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

PROFILE_MANAGER_SCHEMA_VERSION = "3.0"

PROFILE_DDL = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL,
    profile_type TEXT,
    description TEXT,
    created_at TEXT,
    updated_at TEXT,
    is_active INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS profile_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    setting_key TEXT NOT NULL,
    setting_value TEXT,
    FOREIGN KEY(profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE,
    UNIQUE(profile_id, setting_key)
);

CREATE TABLE IF NOT EXISTS profile_strategy_allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    strategy_code TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    target_weight REAL,
    FOREIGN KEY(profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE,
    UNIQUE(profile_id, strategy_code)
);

CREATE TABLE IF NOT EXISTS profile_risk_settings (
    profile_id TEXT PRIMARY KEY,
    daily_dd_limit REAL,
    total_dd_limit REAL,
    target_profit REAL,
    profit_cushion_trigger REAL,
    profit_cushion_multiplier REAL,
    equity_trail_enabled INTEGER,
    equity_trail_trigger REAL,
    equity_trail_distance REAL,
    max_concurrent_positions INTEGER,
    FOREIGN KEY(profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS profile_execution_settings (
    profile_id TEXT PRIMARY KEY,
    spread_model TEXT,
    avg_spread_pips REAL,
    commission_per_lot REAL,
    slippage_pips REAL,
    leverage REAL,
    broker_name TEXT,
    FOREIGN KEY(profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_profile_settings_profile ON profile_settings(profile_id);
CREATE INDEX IF NOT EXISTS idx_profile_alloc_profile ON profile_strategy_allocations(profile_id);
"""

DASHBOARD_STRATEGY_CODES: tuple[str, ...] = ("LSFC", "DBBS", "DiNapoli", "VAMR", "SMRS")

SETUP_TYPE_BY_STRATEGY_CODE: dict[str, str] = {
    "LSFC": "LONDON_SWEEP_FAILURE_CONTINUATION",
    "DBBS": "DBBS",
    "DiNapoli": "DINAPOLI_STRUCTURE",
    "VAMR": "VAMR",
    "SMRS": "SMRS",
}

DEFAULT_PROFILE_SPECS: list[dict[str, Any]] = [
    {
        "profile_id": "PROP_FINTOKEI",
        "profile_name": "Fintokei Prop Challenge",
        "profile_type": "prop",
        "description": "Fintokei 100K challenge — 8% target, profit cushion + equity trail ON",
        "is_active": 1,
        "risk": {
            "daily_dd_limit": 4.5,
            "total_dd_limit": 8.5,
            "target_profit": 8.0,
            "profit_cushion_trigger": 4.0,
            "profit_cushion_multiplier": 0.5,
            "equity_trail_enabled": 1,
            "equity_trail_trigger": 6.0,
            "equity_trail_distance": 1.5,
            "max_concurrent_positions": 5,
        },
        "execution": {
            "spread_model": "variable",
            "avg_spread_pips": 1.2,
            "commission_per_lot": 6.0,
            "slippage_pips": 0.3,
            "leverage": 100.0,
            "broker_name": "Fintokei",
        },
        "settings": {
            "starting_equity": "100000",
            "profile_key": "challenge",
            "profit_cushion_enabled": "1",
        },
        "allocations": {
            "LSFC": 0.20,
            "DBBS": 0.20,
            "DiNapoli": 0.20,
            "VAMR": 0.20,
            "SMRS": 0.20,
        },
    },
    {
        "profile_id": "PROP_FTMO",
        "profile_name": "FTMO Prop Challenge",
        "profile_type": "prop",
        "description": "FTMO 100K — 10% target, 5% daily / 10% total DD",
        "is_active": 0,
        "risk": {
            "daily_dd_limit": 5.0,
            "total_dd_limit": 10.0,
            "target_profit": 10.0,
            "profit_cushion_trigger": 5.0,
            "profit_cushion_multiplier": 0.5,
            "equity_trail_enabled": 1,
            "equity_trail_trigger": 7.0,
            "equity_trail_distance": 2.0,
            "max_concurrent_positions": 5,
        },
        "execution": {
            "spread_model": "variable",
            "avg_spread_pips": 1.0,
            "commission_per_lot": 0.0,
            "slippage_pips": 0.2,
            "leverage": 100.0,
            "broker_name": "FTMO",
        },
        "settings": {
            "starting_equity": "100000",
            "profile_key": "challenge",
            "profit_cushion_enabled": "1",
        },
        "allocations": {
            "LSFC": 0.25,
            "DBBS": 0.25,
            "DiNapoli": 0.25,
            "VAMR": 0.25,
            "SMRS": 0.0,
        },
    },
    {
        "profile_id": "PERSONAL_CONSERVATIVE",
        "profile_name": "Personal Conservative",
        "profile_type": "personal",
        "description": "Self-funded conservative — 5% max DD, equity trail ON",
        "is_active": 0,
        "risk": {
            "daily_dd_limit": 2.5,
            "total_dd_limit": 5.0,
            "target_profit": None,
            "profit_cushion_trigger": None,
            "profit_cushion_multiplier": None,
            "equity_trail_enabled": 1,
            "equity_trail_trigger": 3.0,
            "equity_trail_distance": 1.0,
            "max_concurrent_positions": 3,
        },
        "execution": {
            "spread_model": "fixed",
            "avg_spread_pips": 1.0,
            "commission_per_lot": 3.5,
            "slippage_pips": 0.1,
            "leverage": 30.0,
            "broker_name": "Personal",
        },
        "settings": {
            "starting_equity": "50000",
            "profile_key": "funded",
            "profit_cushion_enabled": "0",
        },
        "allocations": {
            "LSFC": 0.30,
            "DBBS": 0.30,
            "DiNapoli": 0.20,
            "VAMR": 0.20,
            "SMRS": 0.0,
        },
    },
    {
        "profile_id": "PERSONAL_AGGRESSIVE",
        "profile_name": "Personal Aggressive",
        "profile_type": "personal",
        "description": "Self-funded aggressive — 20% max DD, cushions OFF",
        "is_active": 0,
        "risk": {
            "daily_dd_limit": 8.0,
            "total_dd_limit": 20.0,
            "target_profit": None,
            "profit_cushion_trigger": None,
            "profit_cushion_multiplier": None,
            "equity_trail_enabled": 0,
            "equity_trail_trigger": None,
            "equity_trail_distance": None,
            "max_concurrent_positions": 8,
        },
        "execution": {
            "spread_model": "variable",
            "avg_spread_pips": 1.5,
            "commission_per_lot": 3.5,
            "slippage_pips": 0.5,
            "leverage": 100.0,
            "broker_name": "Personal",
        },
        "settings": {
            "starting_equity": "50000",
            "profile_key": "funded",
            "profit_cushion_enabled": "0",
        },
        "allocations": {
            "LSFC": 0.20,
            "DBBS": 0.20,
            "DiNapoli": 0.20,
            "VAMR": 0.20,
            "SMRS": 0.20,
        },
    },
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _insert_profile_bundle(conn: sqlite3.Connection, spec: dict[str, Any]) -> None:
    now = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO profiles (
            profile_id, profile_name, profile_type, description,
            created_at, updated_at, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            spec["profile_id"],
            spec["profile_name"],
            spec.get("profile_type"),
            spec.get("description"),
            now,
            now,
            int(spec.get("is_active", 0)),
        ),
    )
    risk = spec["risk"]
    conn.execute(
        """
        INSERT INTO profile_risk_settings (
            profile_id, daily_dd_limit, total_dd_limit, target_profit,
            profit_cushion_trigger, profit_cushion_multiplier,
            equity_trail_enabled, equity_trail_trigger, equity_trail_distance,
            max_concurrent_positions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            spec["profile_id"],
            risk.get("daily_dd_limit"),
            risk.get("total_dd_limit"),
            risk.get("target_profit"),
            risk.get("profit_cushion_trigger"),
            risk.get("profit_cushion_multiplier"),
            risk.get("equity_trail_enabled"),
            risk.get("equity_trail_trigger"),
            risk.get("equity_trail_distance"),
            risk.get("max_concurrent_positions"),
        ),
    )
    execution = spec["execution"]
    conn.execute(
        """
        INSERT INTO profile_execution_settings (
            profile_id, spread_model, avg_spread_pips, commission_per_lot,
            slippage_pips, leverage, broker_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            spec["profile_id"],
            execution.get("spread_model"),
            execution.get("avg_spread_pips"),
            execution.get("commission_per_lot"),
            execution.get("slippage_pips"),
            execution.get("leverage"),
            execution.get("broker_name"),
        ),
    )
    for key, value in (spec.get("settings") or {}).items():
        conn.execute(
            "INSERT INTO profile_settings (profile_id, setting_key, setting_value) VALUES (?, ?, ?)",
            (spec["profile_id"], key, str(value)),
        )
    allocations = spec.get("allocations") or {}
    for code in DASHBOARD_STRATEGY_CODES:
        weight = float(allocations.get(code, 0.0))
        enabled = 1 if weight > 0.0 else 0
        conn.execute(
            """
            INSERT INTO profile_strategy_allocations (
                profile_id, strategy_code, enabled, target_weight
            ) VALUES (?, ?, ?, ?)
            """,
            (spec["profile_id"], code, enabled, weight),
        )


def seed_default_profiles(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT COUNT(*) AS c FROM profiles").fetchone()
    if row and int(row[0]) > 0:
        return
    for spec in DEFAULT_PROFILE_SPECS:
        _insert_profile_bundle(conn, spec)


def apply_profile_manager_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(PROFILE_DDL)
    seed_default_profiles(conn)
    conn.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES ('profile_manager_schema_version', ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (PROFILE_MANAGER_SCHEMA_VERSION, _utc_now_iso()),
    )
