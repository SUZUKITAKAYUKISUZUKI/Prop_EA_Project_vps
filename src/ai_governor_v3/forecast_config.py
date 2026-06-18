"""Configurable forecast horizons for AGE v3."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_FORECAST_DAYS: tuple[int, ...] = (30, 60, 90)
DETERIORATION_THRESHOLD_PCT = 10.0
RISK_BUDGET_EXHAUSTION_THRESHOLD = 15.0
CACHE_TTL_KEY_PREFIX = "age_v3_forecast"


@dataclass(frozen=True)
class ForecastConfig:
    forecast_days: tuple[int, ...] = DEFAULT_FORECAST_DAYS
    deterioration_threshold_pct: float = DETERIORATION_THRESHOLD_PCT
    risk_budget_exhaustion_threshold: float = RISK_BUDGET_EXHAUSTION_THRESHOLD
    use_cache: bool = True

    def horizons_label(self) -> str:
        return ",".join(str(d) for d in self.forecast_days)


DEFAULT_CONFIG = ForecastConfig()
