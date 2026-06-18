"""Stage-aware allocation rules for PAAE integration."""
from __future__ import annotations

from typing import Any

from src.adaptive_allocation.allocation_constraints import normalize_weights
from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES
from src.strategy_lifecycle.lifecycle_stages import (
    CORE_MIN_ALLOCATION,
    PAAE_ELIGIBLE_STAGES,
    STAGE_ALLOCATION,
    STAGE_CANDIDATE,
    STAGE_CORE,
    STAGE_INCUBATION,
    STAGE_PRODUCTION,
    STAGE_RECOVERY,
    STAGE_RETIRED,
)


class LifecycleManager:
    """Apply lifecycle stage constraints to portfolio weights."""

    def stage_map(self, registry: list[dict[str, Any]]) -> dict[str, str]:
        return {str(r["strategy_id"]): str(r["current_stage"]) for r in registry}

    def core_strategies(self, registry: list[dict[str, Any]]) -> set[str]:
        core: set[str] = set()
        for row in registry:
            stage = str(row.get("current_stage") or "").upper()
            if stage == STAGE_CORE or int(row.get("core_strategy") or 0) == 1:
                core.add(str(row["strategy_id"]))
        return core

    def allocation_for_stage(self, stage: str) -> float | None:
        return STAGE_ALLOCATION.get(stage.upper())

    def apply_stage_allocations(
        self,
        weights: dict[str, float],
        stages: dict[str, str],
        *,
        registry: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        adjusted: dict[str, float] = {}
        reasons: dict[str, str] = {}
        fixed_total = 0.0
        paae_managed: dict[str, float] = {}
        core_codes = self.core_strategies(registry or [])

        for code in DASHBOARD_STRATEGY_CODES:
            stage = stages.get(code, STAGE_INCUBATION).upper()
            if stage in {STAGE_INCUBATION, STAGE_RETIRED}:
                adjusted[code] = 0.0
                reasons[code] = f"{stage.lower()}_zero_allocation"
            elif stage in {STAGE_CANDIDATE, STAGE_RECOVERY}:
                adjusted[code] = float(STAGE_ALLOCATION.get(stage) or 0.05)
                fixed_total += adjusted[code]
                reasons[code] = f"{stage.lower()}_fixed_5pct"
            elif stage in {STAGE_PRODUCTION, STAGE_CORE}:
                paae_managed[code] = max(0.0, float(weights.get(code, 0.0)))
                reasons[code] = "core_paae_min_10pct" if stage == STAGE_CORE else "production_paae_managed"
            else:
                adjusted[code] = 0.0
                reasons[code] = "unknown_stage_zero"

        remaining = max(0.0, 1.0 - fixed_total)
        if paae_managed:
            total = sum(paae_managed.values()) or 1.0
            for code, weight in paae_managed.items():
                adjusted[code] = round(weight / total * remaining, 4)

        for code in core_codes:
            if code in adjusted:
                adjusted[code] = max(adjusted[code], CORE_MIN_ALLOCATION)

        if not any(v > 0 for v in adjusted.values()):
            return {code: 0.0 for code in DASHBOARD_STRATEGY_CODES}, reasons
        return normalize_weights(adjusted), reasons

    def filter_paae_candidates(self, stages: dict[str, str]) -> set[str]:
        disabled: set[str] = set()
        for code in DASHBOARD_STRATEGY_CODES:
            stage = stages.get(code, STAGE_INCUBATION).upper()
            if stage not in PAAE_ELIGIBLE_STAGES:
                disabled.add(code)
        return disabled

    def eligible_for_adaptive_allocation(self, strategy_id: str, stage: str) -> bool:
        return stage.upper() in PAAE_ELIGIBLE_STAGES

    def core_min_weights(self, registry: list[dict[str, Any]]) -> dict[str, float]:
        return {code: CORE_MIN_ALLOCATION for code in self.core_strategies(registry)}
