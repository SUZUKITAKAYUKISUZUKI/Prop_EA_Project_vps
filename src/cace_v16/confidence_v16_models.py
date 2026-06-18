"""Constants and category helpers for CACE v1.6."""

from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_V16_DECAY = "cace:v16:decay:{profile_id}"
CACHE_V16_CONSENSUS = "cace:v16:consensus:{profile_id}"
CACHE_V16_INTELLIGENCE = "cace:v16:intelligence:{profile_id}"

FORECAST_HORIZONS = ("today", "30d", "60d", "90d", "180d")

DURABILITY_CATEGORIES: tuple[tuple[float, str], ...] = (
    (90.0, "EXTREME"),
    (80.0, "VERY_HIGH"),
    (65.0, "HIGH"),
    (50.0, "MODERATE"),
    (35.0, "LOW"),
    (0.0, "VERY_LOW"),
)

CONSENSUS_CATEGORIES: tuple[tuple[float, str], ...] = (
    (95.0, "EXTREME"),
    (85.0, "VERY_HIGH"),
    (70.0, "HIGH"),
    (55.0, "MODERATE"),
    (40.0, "LOW"),
    (0.0, "VERY_LOW"),
)


def durability_category(score: float) -> str:
    for threshold, label in DURABILITY_CATEGORIES:
        if score >= threshold:
            return label
    return "VERY_LOW"


def consensus_category(score: float) -> str:
    for threshold, label in CONSENSUS_CATEGORIES:
        if score >= threshold:
            return label
    return "VERY_LOW"


def half_life_from_stability(stability: float) -> int:
    if stability >= 90:
        return 120
    if stability >= 80:
        return 90
    if stability >= 70:
        return 60
    if stability >= 60:
        return 30
    return 14
