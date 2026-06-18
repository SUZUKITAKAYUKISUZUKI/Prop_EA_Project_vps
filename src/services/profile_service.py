"""Profile Manager service layer — active profile context and config loaders."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from core.prop_profiles import PropProfile
from src.database.profile_migrations import SETUP_TYPE_BY_STRATEGY_CODE
from src.repositories.profile_repository import ProfileRepository


@dataclass
class ProfileContext:
    profile_id: str
    profile_name: str
    profile_type: str | None = None
    description: str | None = None
    target_profit: float | None = None
    daily_dd_limit: float | None = None
    total_dd_limit: float | None = None
    profit_cushion_trigger: float | None = None
    profit_cushion_multiplier: float | None = None
    profit_cushion_enabled: bool = False
    equity_trail_enabled: bool = False
    equity_trail_trigger: float | None = None
    equity_trail_distance: float | None = None
    max_concurrent_positions: int | None = None
    strategy_allocations: dict[str, float] = field(default_factory=dict)
    strategy_enabled: dict[str, bool] = field(default_factory=dict)
    execution_settings: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ProfileContext:
        risk = record.get("risk") or {}
        execution = record.get("execution") or {}
        settings = record.get("settings") or {}
        profit_cushion_enabled = str(settings.get("profit_cushion_enabled", "0")).strip() in (
            "1",
            "true",
            "yes",
            "on",
        )
        return cls(
            profile_id=str(record["profile_id"]),
            profile_name=str(record.get("profile_name") or record["profile_id"]),
            profile_type=record.get("profile_type"),
            description=record.get("description"),
            target_profit=_maybe_float(risk.get("target_profit")),
            daily_dd_limit=_maybe_float(risk.get("daily_dd_limit")),
            total_dd_limit=_maybe_float(risk.get("total_dd_limit")),
            profit_cushion_trigger=_maybe_float(risk.get("profit_cushion_trigger")),
            profit_cushion_multiplier=_maybe_float(risk.get("profit_cushion_multiplier")),
            profit_cushion_enabled=profit_cushion_enabled,
            equity_trail_enabled=bool(risk.get("equity_trail_enabled")),
            equity_trail_trigger=_maybe_float(risk.get("equity_trail_trigger")),
            equity_trail_distance=_maybe_float(risk.get("equity_trail_distance")),
            max_concurrent_positions=_maybe_int(risk.get("max_concurrent_positions")),
            strategy_allocations=dict(record.get("strategy_allocations") or {}),
            strategy_enabled=dict(record.get("strategy_enabled") or {}),
            execution_settings={
                "spread_model": execution.get("spread_model"),
                "avg_spread_pips": _maybe_float(execution.get("avg_spread_pips")),
                "commission_per_lot": _maybe_float(execution.get("commission_per_lot")),
                "slippage_pips": _maybe_float(execution.get("slippage_pips")),
                "leverage": _maybe_float(execution.get("leverage")),
                "broker_name": execution.get("broker_name"),
            },
            settings={str(k): str(v) for k, v in settings.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "profile_type": self.profile_type,
            "description": self.description,
            "target_profit": self.target_profit,
            "daily_dd_limit": self.daily_dd_limit,
            "total_dd_limit": self.total_dd_limit,
            "profit_cushion_trigger": self.profit_cushion_trigger,
            "profit_cushion_multiplier": self.profit_cushion_multiplier,
            "profit_cushion_enabled": self.profit_cushion_enabled,
            "equity_trail_enabled": self.equity_trail_enabled,
            "equity_trail_trigger": self.equity_trail_trigger,
            "equity_trail_distance": self.equity_trail_distance,
            "max_concurrent_positions": self.max_concurrent_positions,
            "strategy_allocations": self.strategy_allocations,
            "strategy_enabled": self.strategy_enabled,
            "execution_settings": self.execution_settings,
            "settings": self.settings,
        }

    def strategy_allocations_by_setup_type(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for code, weight in self.strategy_allocations.items():
            setup = SETUP_TYPE_BY_STRATEGY_CODE.get(code, code)
            if self.strategy_enabled.get(code, weight > 0.0):
                out[setup] = float(weight)
        return out

    def enabled_setup_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.strategy_allocations_by_setup_type().keys()))

    def to_prop_profile(self) -> PropProfile:
        starting_equity = float(self.settings.get("starting_equity") or 100_000.0)
        profile_key = str(self.settings.get("profile_key") or "challenge")
        return PropProfile(
            name=self.profile_id,
            target_profit=float(self.target_profit or 0.0),
            daily_dd_limit=float(self.daily_dd_limit or 5.0),
            total_dd_limit=float(self.total_dd_limit or 10.0),
            max_days=int(float(self.settings.get("max_days") or 0)),
            starting_equity=starting_equity,
            profile_key=profile_key,
        )


_active_context: ProfileContext | None = None


def get_profile_context() -> ProfileContext:
    global _active_context
    if _active_context is None:
        _active_context = ProfileService().load_active_profile()
    return _active_context


def refresh_profile_context() -> ProfileContext:
    global _active_context
    _active_context = ProfileService().load_active_profile()
    return _active_context


def clear_profile_context_cache() -> None:
    global _active_context
    _active_context = None


class ProfileService:
    def __init__(self, repo: ProfileRepository | None = None) -> None:
        self._repo = repo or ProfileRepository()
        self._owns_repo = repo is None

    def close(self) -> None:
        if self._owns_repo:
            self._repo.close()

    def list_profiles(self) -> list[dict[str, Any]]:
        return self._repo.list_profiles()

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        record = self._repo.get_profile(profile_id)
        if record is None:
            raise KeyError(f"Unknown profile: {profile_id}")
        return record

    def load_active_profile(self) -> ProfileContext:
        record = self._repo.get_active_profile()
        if record is None:
            raise RuntimeError("No profiles configured — run database migration")
        return ProfileContext.from_record(record)

    def load_risk_config(self, profile_id: str | None = None) -> dict[str, Any]:
        ctx = self._resolve(profile_id)
        return {
            "profile_id": ctx.profile_id,
            "profile_name": ctx.profile_name,
            "daily_dd_limit": ctx.daily_dd_limit,
            "total_dd_limit": ctx.total_dd_limit,
            "target_profit": ctx.target_profit,
            "profit_cushion_trigger": ctx.profit_cushion_trigger,
            "profit_cushion_multiplier": ctx.profit_cushion_multiplier,
            "profit_cushion_enabled": ctx.profit_cushion_enabled,
            "equity_trail_enabled": ctx.equity_trail_enabled,
            "equity_trail_trigger": ctx.equity_trail_trigger,
            "equity_trail_distance": ctx.equity_trail_distance,
            "max_concurrent_positions": ctx.max_concurrent_positions,
        }

    def load_execution_config(self, profile_id: str | None = None) -> dict[str, Any]:
        ctx = self._resolve(profile_id)
        return {
            "profile_id": ctx.profile_id,
            "profile_name": ctx.profile_name,
            **ctx.execution_settings,
        }

    def load_allocation(self, profile_id: str | None = None) -> dict[str, float]:
        ctx = self._resolve(profile_id)
        return dict(ctx.strategy_allocations_by_setup_type())

    def apply_profile(self, profile_id: str | None = None) -> ProfileContext:
        if profile_id is not None:
            record = self._repo.activate_profile(profile_id)
            ctx = ProfileContext.from_record(record)
        else:
            ctx = self.load_active_profile()
        os.environ["ACTIVE_PROFILE_ID"] = ctx.profile_id
        os.environ["ACTIVE_PROFILE_NAME"] = ctx.profile_name
        if ctx.daily_dd_limit is not None:
            os.environ["PROP_DAILY_DD_LIMIT"] = str(ctx.daily_dd_limit)
        if ctx.total_dd_limit is not None:
            os.environ["PROP_TOTAL_DD_LIMIT"] = str(ctx.total_dd_limit)
        if ctx.target_profit is not None:
            os.environ["PROP_TARGET_PROFIT"] = str(ctx.target_profit)
        weights = ctx.strategy_allocations_by_setup_type()
        if weights:
            import json

            os.environ["PORTFOLIO_STRATEGY_WEIGHTS"] = json.dumps(weights, sort_keys=True)
        os.environ["PROFIT_CUSHION_ENABLED"] = "1" if ctx.profit_cushion_enabled else "0"
        settings = ctx.settings or {}
        if "allocation_enabled" in settings:
            os.environ["PORTFOLIO_ALLOCATION_ENABLED"] = settings["allocation_enabled"]
        if settings.get("portfolio_weight_mode"):
            os.environ["PORTFOLIO_WEIGHT_MODE"] = str(settings["portfolio_weight_mode"])
        if settings.get("allocation_source"):
            os.environ["PORTFOLIO_ALLOCATION_SOURCE"] = str(settings["allocation_source"])
        if settings.get("bayes_threshold"):
            os.environ["PROFILE_BAYES_THRESHOLD"] = str(settings["bayes_threshold"])
        if settings.get("sizing_model"):
            os.environ["SIZING_MODEL"] = str(settings["sizing_model"])
        if settings.get("risk_multiplier"):
            os.environ["PORTFOLIO_RISK_MULTIPLIER"] = str(settings["risk_multiplier"])
        if settings.get("recommended_objective"):
            os.environ["RECOMMENDED_OBJECTIVE"] = str(settings["recommended_objective"])
        global _active_context
        _active_context = ctx
        return ctx

    def resolve_profile_from_state(self, state: str) -> str:
        from src.profile_manager.resolver import resolve_profile_from_state

        return resolve_profile_from_state(state)

    def create_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = self._repo.create_profile(payload)
        clear_profile_context_cache()
        return record

    def save_profile(self, profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["profile_id"] = profile_id
        record = self._repo.save_profile(profile_id, payload)
        clear_profile_context_cache()
        active = self._repo.get_active_profile()
        if active and active["profile_id"] == profile_id:
            refresh_profile_context()
        return record

    def delete_profile(self, profile_id: str) -> None:
        self._repo.delete_profile(profile_id)
        clear_profile_context_cache()
        refresh_profile_context()

    def clone_profile(
        self,
        profile_id: str,
        new_profile_id: str,
        *,
        new_name: str | None = None,
    ) -> dict[str, Any]:
        record = self._repo.clone_profile(profile_id, new_profile_id, new_name=new_name)
        clear_profile_context_cache()
        return record

    def activate_profile(self, profile_id: str) -> ProfileContext:
        return self.apply_profile(profile_id)

    def _resolve(self, profile_id: str | None) -> ProfileContext:
        if profile_id is None:
            return self.load_active_profile()
        return ProfileContext.from_record(self.get_profile(profile_id))


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
