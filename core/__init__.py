"""Core package — PET (live-safe). PFOO symbols are lazy-loaded on demand."""

from __future__ import annotations

from typing import Any

from core.portfolio_equity_trail import (
    PetDecision,
    PetRuntimeState,
    PortfolioEquityTrail,
    apply_pet_to_trade_signal,
    default_pet_enabled_for_profile,
    evaluate_pet,
    is_pet_enabled,
    load_pet_config,
    run_pet_monte_carlo_validation,
)

__all__ = [
    "PropOptimizer",
    "PFOOResult",
    "run_pfoo",
    "PetDecision",
    "PetRuntimeState",
    "PortfolioEquityTrail",
    "apply_pet_to_trade_signal",
    "default_pet_enabled_for_profile",
    "evaluate_pet",
    "is_pet_enabled",
    "load_pet_config",
    "run_pet_monte_carlo_validation",
]

_PFOO_EXPORTS = {
    "PropOptimizer": "PropOptimizer",
    "PFOOResult": "PFOOResult",
    "run_pfoo": "run_pfoo",
}


def __getattr__(name: str) -> Any:
    if name in _PFOO_EXPORTS:
        from core import prop_optimizer as mod

        return getattr(mod, _PFOO_EXPORTS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
