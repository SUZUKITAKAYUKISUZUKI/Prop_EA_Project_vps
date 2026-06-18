"""SQLite repository for Profile Manager (Phase 3.0)."""
from __future__ import annotations

import sqlite3
from copy import deepcopy
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES, DEFAULT_PROFILE_SPECS
from src.repositories.base import create_default_db_manager


class ProfileRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.portfolio

    def list_profiles(self) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT profile_id, profile_name, profile_type, description,
                   created_at, updated_at, is_active
            FROM profiles
            ORDER BY profile_name
            """,
        )
        return [dict(row) for row in rows]

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT profile_id, profile_name, profile_type, description,
                   created_at, updated_at, is_active
            FROM profiles WHERE profile_id=?
            """,
            (profile_id,),
            one=True,
        )
        if row is None:
            return None
        return self._hydrate_profile(dict(row))

    def get_active_profile(self) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT profile_id, profile_name, profile_type, description,
                   created_at, updated_at, is_active
            FROM profiles WHERE is_active=1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            one=True,
        )
        if row is None:
            row = self._db.query(
                """
                SELECT profile_id, profile_name, profile_type, description,
                       created_at, updated_at, is_active
                FROM profiles
                ORDER BY profile_id
                LIMIT 1
                """,
                one=True,
            )
        if row is None:
            return None
        return self._hydrate_profile(dict(row))

    def create_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(payload["profile_id"]).strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        existing = self.get_profile(profile_id)
        if existing is not None:
            raise ValueError(f"Profile already exists: {profile_id}")
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO profiles (
                profile_id, profile_name, profile_type, description,
                created_at, updated_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                str(payload.get("profile_name") or profile_id),
                payload.get("profile_type"),
                payload.get("description"),
                now,
                now,
                int(payload.get("is_active") or 0),
            ),
        )
        self._write_related(profile_id, payload)
        self.conn.commit()
        result = self.get_profile(profile_id)
        assert result is not None
        return result

    def save_profile(self, profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self._db.query("SELECT profile_id FROM profiles WHERE profile_id=?", (profile_id,), one=True)
        if row is None:
            raise KeyError(f"Unknown profile: {profile_id}")
        now = utc_now_iso()
        self.conn.execute(
            """
            UPDATE profiles
            SET profile_name=?, profile_type=?, description=?, updated_at=?
            WHERE profile_id=?
            """,
            (
                str(payload.get("profile_name") or profile_id),
                payload.get("profile_type"),
                payload.get("description"),
                now,
                profile_id,
            ),
        )
        self._delete_related(profile_id)
        self._write_related(profile_id, payload)
        self.conn.commit()
        result = self.get_profile(profile_id)
        assert result is not None
        return result

    def delete_profile(self, profile_id: str) -> None:
        active = self.get_active_profile()
        if active and active["profile_id"] == profile_id:
            raise ValueError("Cannot delete the active profile")
        count = self._db.query("SELECT COUNT(*) AS c FROM profiles", one=True)
        if count and int(count["c"]) <= 1:
            raise ValueError("Cannot delete the last profile")
        self.conn.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
        self.conn.commit()

    def clone_profile(self, profile_id: str, new_profile_id: str, *, new_name: str | None = None) -> dict[str, Any]:
        source = self.get_profile(profile_id)
        if source is None:
            raise KeyError(f"Unknown profile: {profile_id}")
        if self.get_profile(new_profile_id) is not None:
            raise ValueError(f"Profile already exists: {new_profile_id}")
        payload = deepcopy(source)
        payload["profile_id"] = new_profile_id
        payload["profile_name"] = new_name or f"{source['profile_name']} (Copy)"
        payload["is_active"] = 0
        return self.create_profile(payload)

    def activate_profile(self, profile_id: str) -> dict[str, Any]:
        if self.get_profile(profile_id) is None:
            raise KeyError(f"Unknown profile: {profile_id}")
        now = utc_now_iso()
        self.conn.execute("UPDATE profiles SET is_active=0")
        self.conn.execute(
            "UPDATE profiles SET is_active=1, updated_at=? WHERE profile_id=?",
            (now, profile_id),
        )
        self.conn.commit()
        result = self.get_profile(profile_id)
        assert result is not None
        return result

    def _hydrate_profile(self, base: dict[str, Any]) -> dict[str, Any]:
        profile_id = base["profile_id"]
        risk = self._db.query(
            "SELECT * FROM profile_risk_settings WHERE profile_id=?",
            (profile_id,),
            one=True,
        )
        execution = self._db.query(
            "SELECT * FROM profile_execution_settings WHERE profile_id=?",
            (profile_id,),
            one=True,
        )
        settings_rows = self._db.query(
            "SELECT setting_key, setting_value FROM profile_settings WHERE profile_id=?",
            (profile_id,),
        )
        alloc_rows = self._db.query(
            """
            SELECT strategy_code, enabled, target_weight
            FROM profile_strategy_allocations
            WHERE profile_id=?
            ORDER BY strategy_code
            """,
            (profile_id,),
        )
        settings = {str(r["setting_key"]): r["setting_value"] for r in settings_rows}
        allocations = [
            {
                "strategy_code": r["strategy_code"],
                "enabled": bool(r["enabled"]),
                "target_weight": float(r["target_weight"] or 0.0),
            }
            for r in alloc_rows
        ]
        strategy_allocations = {
            row["strategy_code"]: row["target_weight"] for row in allocations if row["enabled"]
        }
        strategy_enabled = {row["strategy_code"]: row["enabled"] for row in allocations}
        out = dict(base)
        out["is_active"] = bool(base.get("is_active"))
        out["risk"] = dict(risk) if risk else {}
        out["execution"] = dict(execution) if execution else {}
        out["settings"] = settings
        out["allocations"] = allocations
        out["strategy_allocations"] = strategy_allocations
        out["strategy_enabled"] = strategy_enabled
        return out

    def _delete_related(self, profile_id: str) -> None:
        self.conn.execute("DELETE FROM profile_settings WHERE profile_id=?", (profile_id,))
        self.conn.execute(
            "DELETE FROM profile_strategy_allocations WHERE profile_id=?",
            (profile_id,),
        )
        self.conn.execute("DELETE FROM profile_risk_settings WHERE profile_id=?", (profile_id,))
        self.conn.execute(
            "DELETE FROM profile_execution_settings WHERE profile_id=?",
            (profile_id,),
        )

    def _write_related(self, profile_id: str, payload: dict[str, Any]) -> None:
        risk = payload.get("risk") or {}
        self.conn.execute(
            """
            INSERT INTO profile_risk_settings (
                profile_id, daily_dd_limit, total_dd_limit, target_profit,
                profit_cushion_trigger, profit_cushion_multiplier,
                equity_trail_enabled, equity_trail_trigger, equity_trail_distance,
                max_concurrent_positions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                _float_or_none(risk.get("daily_dd_limit")),
                _float_or_none(risk.get("total_dd_limit")),
                _float_or_none(risk.get("target_profit")),
                _float_or_none(risk.get("profit_cushion_trigger")),
                _float_or_none(risk.get("profit_cushion_multiplier")),
                _int01(risk.get("equity_trail_enabled")),
                _float_or_none(risk.get("equity_trail_trigger")),
                _float_or_none(risk.get("equity_trail_distance")),
                _int_or_none(risk.get("max_concurrent_positions")),
            ),
        )
        execution = payload.get("execution") or {}
        self.conn.execute(
            """
            INSERT INTO profile_execution_settings (
                profile_id, spread_model, avg_spread_pips, commission_per_lot,
                slippage_pips, leverage, broker_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                execution.get("spread_model"),
                _float_or_none(execution.get("avg_spread_pips")),
                _float_or_none(execution.get("commission_per_lot")),
                _float_or_none(execution.get("slippage_pips")),
                _float_or_none(execution.get("leverage")),
                execution.get("broker_name"),
            ),
        )
        settings = payload.get("settings") or {}
        for key, value in settings.items():
            self.conn.execute(
                "INSERT INTO profile_settings (profile_id, setting_key, setting_value) VALUES (?, ?, ?)",
                (profile_id, str(key), str(value)),
            )
        alloc_map = _normalize_allocations(payload)
        for code in DASHBOARD_STRATEGY_CODES:
            weight = float(alloc_map.get(code, 0.0))
            enabled = payload.get("strategy_enabled", {}).get(code)
            if enabled is None:
                enabled = weight > 0.0
            self.conn.execute(
                """
                INSERT INTO profile_strategy_allocations (
                    profile_id, strategy_code, enabled, target_weight
                ) VALUES (?, ?, ?, ?)
                """,
                (profile_id, code, 1 if enabled else 0, weight),
            )

    @staticmethod
    def default_profile_template(profile_id: str, profile_name: str) -> dict[str, Any]:
        base = deepcopy(DEFAULT_PROFILE_SPECS[0])
        base["profile_id"] = profile_id
        base["profile_name"] = profile_name
        base["is_active"] = 0
        return base


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int01(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return 1 if bool(value) else 0


def _normalize_allocations(payload: dict[str, Any]) -> dict[str, float]:
    if payload.get("strategy_allocations"):
        return {str(k): float(v) for k, v in payload["strategy_allocations"].items()}
    if payload.get("allocations") and isinstance(payload["allocations"], dict):
        if all(isinstance(v, (int, float)) for v in payload["allocations"].values()):
            return {str(k): float(v) for k, v in payload["allocations"].items()}
    out: dict[str, float] = {}
    for row in payload.get("allocations") or []:
        if isinstance(row, dict):
            code = str(row.get("strategy_code") or row.get("code") or "")
            if code:
                out[code] = float(row.get("target_weight") or 0.0)
    return out
