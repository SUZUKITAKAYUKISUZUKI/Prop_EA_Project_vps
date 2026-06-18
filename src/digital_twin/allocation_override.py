"""Apply allocation overrides to a profile context for what-if simulation."""
from __future__ import annotations

from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES
from src.services.profile_service import ProfileContext


def normalize_allocation(weights: dict[str, float]) -> dict[str, float]:
    positive = {k: max(0.0, float(v)) for k, v in weights.items()}
    for code in DASHBOARD_STRATEGY_CODES:
        positive.setdefault(code, 0.0)
    total = sum(positive.values())
    if total <= 0:
        n = len(DASHBOARD_STRATEGY_CODES) or 1
        return {code: round(1.0 / n, 4) for code in DASHBOARD_STRATEGY_CODES}
    return {k: round(v / total, 4) for k, v in positive.items()}


def apply_allocation_override(
    ctx: ProfileContext,
    weights: dict[str, float],
    *,
    enable_nonzero: bool = True,
) -> ProfileContext:
    normalized = normalize_allocation(weights)
    ctx.strategy_allocations = dict(normalized)
    if enable_nonzero:
        ctx.strategy_enabled = {code: weight > 0 for code, weight in normalized.items()}
    return ctx
