"""Scenario definitions for Portfolio Digital Twin."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.adaptive_allocation.allocation_policy import base_weights_for_state, resolve_account_state
from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES
from src.digital_twin.allocation_override import normalize_allocation
from src.services.profile_service import ProfileContext

SCENARIO_BASELINE = "baseline"
SCENARIO_RECOMMENDED = "recommended"
SCENARIO_CHALLENGE_AGGRESSIVE = "challenge_aggressive"
SCENARIO_FUNDED_GROWTH = "funded_growth"
SCENARIO_RECOVERY_DEFENSIVE = "recovery_defensive"
SCENARIO_CUSTOM = "custom"

BUILTIN_SCENARIOS = (
    SCENARIO_BASELINE,
    SCENARIO_RECOMMENDED,
    SCENARIO_CHALLENGE_AGGRESSIVE,
    SCENARIO_FUNDED_GROWTH,
    SCENARIO_RECOVERY_DEFENSIVE,
    SCENARIO_CUSTOM,
)


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    label: str
    allocation: dict[str, float]
    account_state: str
    profile_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _percent_weights(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    sample = next(iter(raw.values()))
    if sample > 1.5:
        return normalize_allocation({k: float(v) / 100.0 for k, v in raw.items()})
    return normalize_allocation(raw)


def build_scenario(
    scenario_name: str,
    *,
    profile_ctx: ProfileContext,
    paae_report: dict[str, Any] | None = None,
    custom_allocation: dict[str, float] | None = None,
) -> ScenarioDefinition:
    paae_report = paae_report or {}
    account_state = str(
        paae_report.get("account_state")
        or resolve_account_state(profile_ctx.profile_id, profile_ctx.settings)
    ).lower()
    current = normalize_allocation(profile_ctx.strategy_allocations or {})

    if scenario_name == SCENARIO_BASELINE:
        allocation = current
        label = "Current Active Profile"
    elif scenario_name == SCENARIO_RECOMMENDED:
        rec = paae_report.get("recommended_weights") or {}
        allocation = _percent_weights(rec) if rec else current
        label = "PAAE Recommended"
    elif scenario_name == SCENARIO_CHALLENGE_AGGRESSIVE:
        allocation = base_weights_for_state("challenge")
        label = "Challenge Aggressive"
        account_state = "challenge"
    elif scenario_name == SCENARIO_FUNDED_GROWTH:
        allocation = base_weights_for_state("funded")
        label = "Funded Growth"
        account_state = "funded"
    elif scenario_name == SCENARIO_RECOVERY_DEFENSIVE:
        allocation = base_weights_for_state("recovery")
        label = "Recovery Defensive"
        account_state = "recovery"
    elif scenario_name == SCENARIO_CUSTOM:
        if not custom_allocation:
            raise ValueError("custom scenario requires custom_allocation")
        allocation = _percent_weights(custom_allocation)
        label = "Custom Allocation"
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    for code in DASHBOARD_STRATEGY_CODES:
        allocation.setdefault(code, 0.0)

    return ScenarioDefinition(
        name=scenario_name,
        label=label,
        allocation=normalize_allocation(allocation),
        account_state=account_state,
        profile_id=profile_ctx.profile_id,
        metadata={"source": scenario_name},
    )


def build_comparison_set(
    *,
    profile_ctx: ProfileContext,
    paae_report: dict[str, Any] | None = None,
    include: tuple[str, ...] | None = None,
) -> list[ScenarioDefinition]:
    names = include or (
        SCENARIO_BASELINE,
        SCENARIO_RECOMMENDED,
        SCENARIO_CHALLENGE_AGGRESSIVE,
        SCENARIO_FUNDED_GROWTH,
        SCENARIO_RECOVERY_DEFENSIVE,
    )
    return [
        build_scenario(name, profile_ctx=profile_ctx, paae_report=paae_report)
        for name in names
        if name != SCENARIO_CUSTOM
    ]
