"""Auto Switch Engine — migrations and profile seeds."""
from __future__ import annotations

import sqlite3
from typing import Any

from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES, _insert_profile_bundle, _utc_now_iso

AUTO_SWITCH_SCHEMA_VERSION = "1.0"

AUTO_SWITCH_PROFILE_SPECS: list[dict[str, Any]] = [
    {
        "profile_id": "ChallengeAggressive",
        "profile_name": "Challenge Aggressive",
        "profile_type": "prop",
        "description": "Fastest challenge pass — allocation OFF, equal weights, G2 sizing",
        "is_active": 0,
        "risk": {
            "daily_dd_limit": 4.5,
            "total_dd_limit": 8.5,
            "target_profit": 8.0,
            "profit_cushion_trigger": 4.0,
            "profit_cushion_multiplier": 0.65,
            "equity_trail_enabled": 0,
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
            "allocation_enabled": "0",
            "portfolio_weight_mode": "equal",
            "allocation_source": "profile",
            "bayes_threshold": "0.70",
            "sizing_model": "G2",
            "risk_multiplier": "1.0",
            "recommended_objective": "FASTEST PASS",
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
        "profile_id": "FundedBalanced",
        "profile_name": "Funded Balanced",
        "profile_type": "prop",
        "description": "Funded low-DD operation — allocation ON via optimizer",
        "is_active": 0,
        "risk": {
            "daily_dd_limit": 4.5,
            "total_dd_limit": 8.5,
            "target_profit": 8.0,
            "profit_cushion_trigger": 4.0,
            "profit_cushion_multiplier": 0.65,
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
            "profile_key": "funded",
            "profit_cushion_enabled": "1",
            "allocation_enabled": "1",
            "portfolio_weight_mode": "optimizer",
            "allocation_source": "optimizer",
            "bayes_threshold": "0.80",
            "sizing_model": "A",
            "risk_multiplier": "0.75",
            "recommended_objective": "STABILITY FIRST",
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
        "profile_id": "RecoveryDefensive",
        "profile_name": "Recovery Defensive",
        "profile_type": "prop",
        "description": "Survival mode — tight Bayes, low risk multiplier",
        "is_active": 0,
        "risk": {
            "daily_dd_limit": 4.5,
            "total_dd_limit": 8.5,
            "target_profit": 8.0,
            "profit_cushion_trigger": 3.0,
            "profit_cushion_multiplier": 0.50,
            "equity_trail_enabled": 1,
            "equity_trail_trigger": 4.0,
            "equity_trail_distance": 1.0,
            "max_concurrent_positions": 3,
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
            "profile_key": "funded",
            "profit_cushion_enabled": "1",
            "allocation_enabled": "1",
            "portfolio_weight_mode": "optimizer",
            "allocation_source": "optimizer",
            "bayes_threshold": "0.90",
            "sizing_model": "A",
            "risk_multiplier": "0.30",
            "recommended_objective": "CAPITAL PRESERVATION",
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
        "profile_id": "LiveCapitalGrowth",
        "profile_name": "Live Capital Growth",
        "profile_type": "live",
        "description": "Live account capital growth with moderated risk",
        "is_active": 0,
        "risk": {
            "daily_dd_limit": 3.0,
            "total_dd_limit": 6.0,
            "target_profit": None,
            "profit_cushion_trigger": 3.0,
            "profit_cushion_multiplier": 0.60,
            "equity_trail_enabled": 1,
            "equity_trail_trigger": 4.0,
            "equity_trail_distance": 1.0,
            "max_concurrent_positions": 4,
        },
        "execution": {
            "spread_model": "variable",
            "avg_spread_pips": 1.0,
            "commission_per_lot": 3.5,
            "slippage_pips": 0.2,
            "leverage": 50.0,
            "broker_name": "Live",
        },
        "settings": {
            "starting_equity": "100000",
            "profile_key": "funded",
            "profit_cushion_enabled": "1",
            "allocation_enabled": "1",
            "portfolio_weight_mode": "optimizer",
            "allocation_source": "optimizer",
            "bayes_threshold": "0.75",
            "sizing_model": "G2",
            "risk_multiplier": "0.85",
            "recommended_objective": "CAPITAL GROWTH",
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

SWITCH_LOG_DDL = """
CREATE TABLE IF NOT EXISTS profile_switch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    old_profile TEXT,
    new_profile TEXT NOT NULL,
    account_state TEXT NOT NULL,
    reason TEXT,
    equity REAL,
    dd REAL
);

CREATE INDEX IF NOT EXISTS idx_profile_switch_log_ts ON profile_switch_log(timestamp DESC);
"""


def _profile_exists(conn: sqlite3.Connection, profile_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM profiles WHERE profile_id=? LIMIT 1",
        (profile_id,),
    ).fetchone()
    return row is not None


def seed_auto_switch_profiles(conn: sqlite3.Connection) -> None:
    for spec in AUTO_SWITCH_PROFILE_SPECS:
        if _profile_exists(conn, spec["profile_id"]):
            continue
        _insert_profile_bundle(conn, spec)


def apply_auto_switch_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(SWITCH_LOG_DDL)
    seed_auto_switch_profiles(conn)
    conn.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        ("auto_switch_schema_version", AUTO_SWITCH_SCHEMA_VERSION, _utc_now_iso()),
    )
