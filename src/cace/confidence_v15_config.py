"""CACE v1.5 configuration and cache keys."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600

CACHE_V15_PORTFOLIO = "cace:v15:portfolio:{profile_id}"
CACHE_V15_TREND = "cace:v15:trend:{profile_id}"
CACHE_V15_REGIME = "cace:v15:regime:{profile_id}"
CACHE_V15_HISTORY = "cace:v15:history:{profile_id}"

TREND_WINDOWS_DAYS = (7, 30, 90)

REGIME_TYPES = (
    "TRENDING",
    "RANGING",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
    "TRANSITIONAL",
)

REGIME_MODIFIERS: dict[str, float] = {
    "TRENDING": 10.0,
    "RANGING": 0.0,
    "HIGH_VOLATILITY": -8.0,
    "LOW_VOLATILITY": 5.0,
    "TRANSITIONAL": -3.0,
}

BREAKDOWN_KEYS = (
    "historical_reliability",
    "monte_carlo_stability",
    "forecast_stability",
    "portfolio_health",
    "lifecycle_quality",
)
