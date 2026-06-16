"""Phase 5.8 — Endgame Mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.pass_probability import ChallengeState


@dataclass(frozen=True)
class EndgameDecision:
    active: bool
    risk_multiplier: float
    min_bayes_probability: float
    prefer_low_variance: bool
    reason: str


def evaluate_endgame_mode(
    challenge: ChallengeState,
    *,
    target_profit_pct: float,
    config: dict[str, Any] | None = None,
) -> EndgameDecision:
    """
    Trigger when profit progress exceeds 80% of target (configurable).

    Goal: maximize pass probability, not maximize R.
    """
    cfg = (config or {}).get("endgame_mode") or {}
    trigger = float(cfg.get("progress_trigger_pct_of_target", 80.0))
    if target_profit_pct <= 0:
        return EndgameDecision(False, 1.0, 0.0, False, "NORMAL")

    progress_ratio = challenge.profit_progress_percent / target_profit_pct * 100.0
    if progress_ratio < trigger:
        return EndgameDecision(
            active=False,
            risk_multiplier=1.0,
            min_bayes_probability=0.0,
            prefer_low_variance=False,
            reason="NORMAL",
        )

    return EndgameDecision(
        active=True,
        risk_multiplier=float(cfg.get("risk_multiplier", 0.6)),
        min_bayes_probability=float(cfg.get("min_bayes_probability", 0.6)),
        prefer_low_variance=bool(cfg.get("prefer_low_variance", True)),
        reason=f"ENDGAME ({progress_ratio:.0f}% of target reached)",
    )
