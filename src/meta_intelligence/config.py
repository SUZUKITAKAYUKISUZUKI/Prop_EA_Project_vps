"""Configuration constants for Meta Intelligence Engine v1."""
from __future__ import annotations

CACHE_TTL_SECONDS = 3600
CACHE_MIE_TRUST = "mie:trust:{profile_id}"
CACHE_MIE_RANKING = "mie:ranking:{profile_id}"
CACHE_MIE_DRIFT = "mie:drift:{profile_id}"
CACHE_MIE_IMPROVEMENTS = "mie:improvements:{profile_id}"
CACHE_MIE_INTELLIGENCE = "mie:intelligence:{profile_id}"

MODULES = ("PRAE", "PAAE", "PDTS", "SLM", "AGE", "CACE")

TRUST_WEIGHTS = {
    "historical_accuracy": 0.35,
    "calibration": 0.25,
    "stability": 0.20,
    "consensus": 0.10,
    "predictive_reliability": 0.10,
}

TRUST_CATEGORIES: tuple[tuple[float, str], ...] = (
    (95.0, "ELITE"),
    (85.0, "VERY_HIGH"),
    (70.0, "HIGH"),
    (55.0, "MODERATE"),
    (40.0, "LOW"),
    (0.0, "UNTRUSTED"),
)

DRIFT_LOOKBACK_DAYS = 90
DRIFT_WARNING_THRESHOLD = 10.0


def trust_category(score: float) -> str:
    for threshold, label in TRUST_CATEGORIES:
        if score >= threshold:
            return label
    return "UNTRUSTED"
