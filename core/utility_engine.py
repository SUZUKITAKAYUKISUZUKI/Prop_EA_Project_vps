"""Phase 5.2 / 5.5 — Dynamic objective modes and utility function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ObjectiveMode = Literal["SAFE", "BALANCED", "AGGRESSIVE"]


@dataclass(frozen=True)
class UtilityResult:
    utility: float
    pass_component: float
    speed_component: float
    capital_component: float
    mode: str


def mode_weights(config: dict[str, Any], mode: ObjectiveMode) -> dict[str, float]:
    modes = config.get("objective_modes") or {}
    default = config.get("utility_weights") or {
        "pass_probability": 0.6,
        "speed_score": 0.3,
        "capital_preservation": 0.1,
    }
    return dict(modes.get(mode.upper(), default))


def speed_score(expected_pass_days: float, *, reference_days: float = 30.0) -> float:
    """Higher is better — faster pass."""
    if expected_pass_days <= 0:
        return 1.0
    return min(1.0, reference_days / expected_pass_days)


def capital_preservation_score(
    total_dd_used_pct: float,
    *,
    total_dd_limit: float,
) -> float:
    """Higher when more DD headroom remains."""
    if total_dd_limit <= 0:
        return 1.0
    used_ratio = min(1.0, max(0.0, total_dd_used_pct / total_dd_limit))
    return 1.0 - used_ratio


def compute_utility(
    *,
    pass_probability: float,
    expected_pass_days: float,
    total_dd_used_pct: float,
    total_dd_limit: float,
    mode: ObjectiveMode = "BALANCED",
    config: dict[str, Any] | None = None,
) -> UtilityResult:
    """
    Utility =
      w1 * PassProbability
    + w2 * SpeedScore
    + w3 * CapitalPreservation
    """
    cfg = config or {}
    weights = mode_weights(cfg, mode)
    w_pass = float(weights.get("pass_probability", 0.6))
    w_speed = float(weights.get("speed_score", 0.3))
    w_cap = float(weights.get("capital_preservation", 0.1))

    pass_norm = pass_probability / 100.0
    speed = speed_score(expected_pass_days)
    cap = capital_preservation_score(total_dd_used_pct, total_dd_limit=total_dd_limit)

    utility = w_pass * pass_norm + w_speed * speed + w_cap * cap
    return UtilityResult(
        utility=round(utility, 4),
        pass_component=round(w_pass * pass_norm, 4),
        speed_component=round(w_speed * speed, 4),
        capital_component=round(w_cap * cap, 4),
        mode=mode.upper(),
    )
