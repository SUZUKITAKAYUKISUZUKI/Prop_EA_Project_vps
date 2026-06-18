"""Profile Manager dashboard API — JSON-ready dict responses."""
from __future__ import annotations

from typing import Any

from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES
from src.services.profile_service import ProfileService, get_profile_context, refresh_profile_context


def _serialize_profile(record: dict[str, Any]) -> dict[str, Any]:
    risk = record.get("risk") or {}
    execution = record.get("execution") or {}
    return {
        "profile_id": record["profile_id"],
        "profile_name": record.get("profile_name"),
        "profile_type": record.get("profile_type"),
        "description": record.get("description"),
        "is_active": bool(record.get("is_active")),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "risk": {
            "daily_dd_limit": risk.get("daily_dd_limit"),
            "total_dd_limit": risk.get("total_dd_limit"),
            "target_profit": risk.get("target_profit"),
            "profit_cushion_trigger": risk.get("profit_cushion_trigger"),
            "profit_cushion_multiplier": risk.get("profit_cushion_multiplier"),
            "equity_trail_enabled": bool(risk.get("equity_trail_enabled")),
            "equity_trail_trigger": risk.get("equity_trail_trigger"),
            "equity_trail_distance": risk.get("equity_trail_distance"),
            "max_concurrent_positions": risk.get("max_concurrent_positions"),
        },
        "execution": {
            "spread_model": execution.get("spread_model"),
            "avg_spread_pips": execution.get("avg_spread_pips"),
            "commission_per_lot": execution.get("commission_per_lot"),
            "slippage_pips": execution.get("slippage_pips"),
            "leverage": execution.get("leverage"),
            "broker_name": execution.get("broker_name"),
        },
        "settings": record.get("settings") or {},
        "strategy_codes": list(DASHBOARD_STRATEGY_CODES),
        "allocations": record.get("allocations") or [],
        "strategy_allocations": record.get("strategy_allocations") or {},
        "strategy_enabled": record.get("strategy_enabled") or {},
    }


def get_profiles() -> list[dict[str, Any]]:
    svc = ProfileService()
    try:
        return [_serialize_profile(row) for row in svc.list_profiles()]
    finally:
        svc.close()


def get_active_profile() -> dict[str, Any]:
    ctx = get_profile_context()
    return ctx.to_dict()


def get_profile(profile_id: str) -> dict[str, Any]:
    svc = ProfileService()
    try:
        return _serialize_profile(svc.get_profile(profile_id))
    finally:
        svc.close()


def switch_profile(profile_id: str) -> dict[str, Any]:
    svc = ProfileService()
    try:
        ctx = svc.activate_profile(profile_id)
        return ctx.to_dict()
    finally:
        svc.close()


def create_profile(payload: dict[str, Any]) -> dict[str, Any]:
    svc = ProfileService()
    try:
        return _serialize_profile(svc.create_profile(payload))
    finally:
        svc.close()


def clone_profile(profile_id: str, new_profile_id: str, *, new_name: str | None = None) -> dict[str, Any]:
    svc = ProfileService()
    try:
        return _serialize_profile(
            svc.clone_profile(profile_id, new_profile_id, new_name=new_name)
        )
    finally:
        svc.close()


def save_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    svc = ProfileService()
    try:
        return _serialize_profile(svc.save_profile(profile_id, payload))
    finally:
        svc.close()


def delete_profile(profile_id: str) -> dict[str, Any]:
    svc = ProfileService()
    try:
        svc.delete_profile(profile_id)
        refresh_profile_context()
        return {"ok": True, "profile_id": profile_id}
    finally:
        svc.close()
