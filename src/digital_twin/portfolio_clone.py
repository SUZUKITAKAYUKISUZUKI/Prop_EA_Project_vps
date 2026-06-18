"""Clone profile context for scenario simulation without mutating live profile."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.services.profile_service import ProfileContext


def clone_profile_context(ctx: ProfileContext, *, suffix: str = "scenario") -> ProfileContext:
    data = ctx.to_dict()
    clone_id = f"{ctx.profile_id}__{suffix}"
    return ProfileContext.from_record(
        {
            "profile_id": clone_id,
            "profile_name": f"{ctx.profile_name} ({suffix})",
            "profile_type": ctx.profile_type,
            "description": ctx.description,
            "risk": {
                "target_profit": ctx.target_profit,
                "daily_dd_limit": ctx.daily_dd_limit,
                "total_dd_limit": ctx.total_dd_limit,
                "profit_cushion_trigger": ctx.profit_cushion_trigger,
                "profit_cushion_multiplier": ctx.profit_cushion_multiplier,
                "equity_trail_enabled": ctx.equity_trail_enabled,
                "equity_trail_trigger": ctx.equity_trail_trigger,
                "equity_trail_distance": ctx.equity_trail_distance,
                "max_concurrent_positions": ctx.max_concurrent_positions,
            },
            "execution": deepcopy(ctx.execution_settings),
            "settings": deepcopy(ctx.settings),
            "strategy_allocations": deepcopy(ctx.strategy_allocations),
            "strategy_enabled": deepcopy(ctx.strategy_enabled),
        }
    )


def clone_from_record(record: dict[str, Any], *, new_id: str | None = None) -> ProfileContext:
    payload = deepcopy(record)
    if new_id:
        payload["profile_id"] = new_id
    return ProfileContext.from_record(payload)
