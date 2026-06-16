"""PET Phase 5.2 — multi-stage portfolio equity lock fractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PetStage:
    stage: int
    name: str
    min_gain_r: float
    max_gain_r: float | None
    lock_fraction: float


def load_pet_stages(config: dict[str, Any] | None = None) -> tuple[PetStage, ...]:
    cfg = config or {}
    raw = cfg.get("stages") or []
    stages: list[PetStage] = []
    for row in raw:
        stages.append(
            PetStage(
                stage=int(row["stage"]),
                name=str(row["name"]),
                min_gain_r=float(row["min_gain_r"]),
                max_gain_r=None if row.get("max_gain_r") is None else float(row["max_gain_r"]),
                lock_fraction=float(row["lock_fraction"]),
            )
        )
    return tuple(sorted(stages, key=lambda s: s.stage))


def resolve_stage(peak_gain_r: float, stages: tuple[PetStage, ...]) -> PetStage:
    if not stages:
        return PetStage(0, "OFF", 0.0, None, 0.0)
    selected = stages[0]
    for stage in stages:
        if peak_gain_r < stage.min_gain_r:
            break
        if stage.max_gain_r is None or peak_gain_r < stage.max_gain_r:
            return stage
        selected = stage
    return selected


def effective_lock_fraction(
    stage: PetStage,
    *,
    challenge_progress_pct: float,
    challenge_trigger_pct: float,
    challenge_lock_bonus: float,
    endgame_lock_multiplier: float,
    endgame_active: bool,
) -> float:
    lock = stage.lock_fraction
    if challenge_progress_pct >= challenge_trigger_pct:
        lock = min(1.0, lock + challenge_lock_bonus)
    if endgame_active and lock > 0.0:
        lock = min(1.0, lock * endgame_lock_multiplier)
    return lock


def compute_locked_profit_r(peak_gain_r: float, lock_fraction: float) -> float:
    if peak_gain_r <= 0.0 or lock_fraction <= 0.0:
        return 0.0
    return round(peak_gain_r * lock_fraction, 4)


def execution_buffer_usd(
    total_lot: float,
    *,
    per_lot_usd: float,
    min_net_profit_usd: float,
) -> float:
    if total_lot <= 0.0:
        return min_net_profit_usd
    return per_lot_usd * total_lot + min_net_profit_usd


def compute_protected_equity(
    day_start_equity: float,
    locked_profit_r: float,
    r_unit_usd: float,
    execution_buffer_usd_value: float,
) -> float:
    locked_usd = locked_profit_r * r_unit_usd
    return round(day_start_equity + locked_usd + execution_buffer_usd_value, 2)
