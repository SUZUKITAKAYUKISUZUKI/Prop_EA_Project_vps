"""Configuration for CACE v1.7 calibration intelligence."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_V17_CALIBRATION = "cace:v17:calibration:{profile_id}"
CACHE_V17_ACCURACY = "cace:v17:accuracy:{profile_id}"
CACHE_V17_RELIABILITY = "cace:v17:reliability:{profile_id}"
CACHE_V17_LEARNING = "cace:v17:learning:{profile_id}"
CACHE_V17_INTELLIGENCE = "cace:v17:intelligence:{profile_id}"

EVALUATION_HORIZON_DAYS = 30
RELIABILITY_WINDOWS = (30, 90, 180)

MODULE_KEYS = ("PAAE", "PDTS", "SLM", "AGE", "CACE")

DECISION_TYPES = (
    "BENEFIT",
    "DRAWDOWN",
    "RECOVERY",
    "PROFILE",
    "ALLOCATION",
    "RECOMMENDATION",
)

CALIBRATION_CATEGORIES: tuple[tuple[float, str], ...] = (
    (95.0, "PERFECT"),
    (85.0, "EXCELLENT"),
    (75.0, "GOOD"),
    (60.0, "FAIR"),
    (0.0, "POOR"),
)

RELIABILITY_TRENDS = ("IMPROVING", "STABLE", "DEGRADING")

SUCCESS_BENEFIT_THRESHOLD = 0.0
SUCCESS_DD_TOLERANCE = 0.5


def calibration_category(score: float) -> str:
    for threshold, label in CALIBRATION_CATEGORIES:
        if score >= threshold:
            return label
    return "POOR"
