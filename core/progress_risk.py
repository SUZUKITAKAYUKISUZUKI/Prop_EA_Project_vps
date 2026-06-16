"""Phase 5.1 — Progress-Aware Risk Engine."""

from __future__ import annotations

from typing import Any


def progress_risk_multiplier(progress_pct: float, bands: list[dict[str, float]] | None = None) -> float:
    """
    Map challenge profit progress (%) to a risk multiplier.

    Default bands (configurable via pfoo_config.json):
      < 2%  → 0.65
      2–5%  → 1.00
      5–8%  → 1.20
      ≥ 8%  → 0.75  (near target — protect profits)
    """
    if bands is None:
        bands = [
            {"max_progress_pct": 2.0, "risk_multiplier": 0.65},
            {"max_progress_pct": 5.0, "risk_multiplier": 1.0},
            {"max_progress_pct": 8.0, "risk_multiplier": 1.2},
            {"max_progress_pct": 999.0, "risk_multiplier": 0.75},
        ]
    ordered = sorted(bands, key=lambda b: float(b["max_progress_pct"]))
    for band in ordered:
        if progress_pct <= float(band["max_progress_pct"]):
            return float(band["risk_multiplier"])
    return float(ordered[-1]["risk_multiplier"])


def apply_progress_risk_to_lot_factor(
    lot_factor: float,
    progress_pct: float,
    config: dict[str, Any] | None = None,
) -> float:
    bands = (config or {}).get("progress_risk_bands")
    mult = progress_risk_multiplier(progress_pct, bands)
    return round(max(0.0, lot_factor * mult), 6)
