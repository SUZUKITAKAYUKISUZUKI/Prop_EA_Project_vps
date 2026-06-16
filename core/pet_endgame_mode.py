"""PET Phase 5.2 — endgame tightening when challenge progress is high."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PetEndgameDecision:
    active: bool
    lock_multiplier: float
    bayes_threshold_delta: float
    risk_multiplier: float
    reason: str


def evaluate_pet_endgame(
    challenge_progress_pct: float,
    *,
    target_profit_pct: float,
    config: dict[str, Any] | None = None,
) -> PetEndgameDecision:
    cfg = (config or {}).get("endgame_mode") or {}
    trigger = float(cfg.get("progress_trigger_pct", 80.0))
    if target_profit_pct <= 0.0:
        progress_ratio = 0.0
    else:
        progress_ratio = challenge_progress_pct / target_profit_pct * 100.0

    if progress_ratio < trigger:
        return PetEndgameDecision(
            active=False,
            lock_multiplier=1.0,
            bayes_threshold_delta=0.0,
            risk_multiplier=1.0,
            reason="NORMAL",
        )

    return PetEndgameDecision(
        active=True,
        lock_multiplier=float(cfg.get("lock_multiplier", 1.25)),
        bayes_threshold_delta=float(cfg.get("bayes_threshold_delta", 0.05)),
        risk_multiplier=float(cfg.get("risk_multiplier", 0.75)),
        reason=f"PET_ENDGAME ({progress_ratio:.0f}% of target)",
    )
