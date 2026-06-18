"""Configuration for AGE v4 strategic simulation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StrategicConfig:
    forecast_days: tuple[int, ...] = (30, 60, 90, 180)
    use_cache: bool = True
    cache_key_prefix: str = "age_v4_strategic"
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "future_health": 0.25,
            "expected_pf": 0.20,
            "recovery_probability": 0.20,
            "dd_protection": 0.15,
            "risk_budget": 0.10,
            "pass_probability": 0.10,
        }
    )
    primary_horizon_days: int = 90

    def horizons_label(self) -> str:
        return ",".join(f"{d}d" for d in self.forecast_days)

    def horizon_key(self, days: int) -> str:
        return f"{days}d"


DEFAULT_STRATEGIC_CONFIG = StrategicConfig()

RANK_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (95.0, "STRONGLY_RECOMMENDED"),
    (80.0, "RECOMMENDED"),
    (65.0, "ACCEPTABLE"),
    (50.0, "HIGH_RISK"),
    (0.0, "REJECT"),
)
