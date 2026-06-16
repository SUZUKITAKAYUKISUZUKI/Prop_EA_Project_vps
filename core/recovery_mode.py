"""Phase 5.7 — Recovery Mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.pass_probability import ChallengeState


@dataclass(frozen=True)
class RecoveryDecision:
    active: bool
    risk_multiplier: float
    min_bayes_probability: float
    disabled_strategies: tuple[str, ...]
    reason: str


def evaluate_recovery_mode(
    challenge: ChallengeState,
    *,
    total_dd_limit: float,
    weakest_strategies: tuple[str, ...] = (),
    config: dict[str, Any] | None = None,
) -> RecoveryDecision:
    """
    Trigger when total DD used exceeds 50% of firm limit (configurable).

    Actions: reduce risk, disable weakest strategies, raise Bayes threshold.
    """
    cfg = (config or {}).get("recovery_mode") or {}
    trigger_pct = float(cfg.get("dd_used_trigger_pct_of_limit", 50.0))
    used_ratio = (
        challenge.total_dd_used_percent / total_dd_limit * 100.0
        if total_dd_limit > 0
        else 0.0
    )

    if used_ratio < trigger_pct:
        return RecoveryDecision(
            active=False,
            risk_multiplier=1.0,
            min_bayes_probability=0.0,
            disabled_strategies=(),
            reason="NORMAL",
        )

    disable_n = int(cfg.get("disable_weakest_strategy_count", 1))
    disabled = tuple(weakest_strategies[:disable_n])
    return RecoveryDecision(
        active=True,
        risk_multiplier=float(cfg.get("risk_multiplier", 0.5)),
        min_bayes_probability=float(cfg.get("min_bayes_probability", 0.55)),
        disabled_strategies=disabled,
        reason=f"RECOVERY (DD used {used_ratio:.1f}% of limit)",
    )
