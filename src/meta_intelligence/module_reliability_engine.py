"""Predictive reliability per module."""
from __future__ import annotations

from typing import Any

from src.meta_intelligence.config import MODULES


class ModuleReliabilityEngine:
    def evaluate(
        self,
        *,
        cace_v17_report: dict[str, Any] | None = None,
        upstream: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        report = cace_v17_report or {}
        reliability = report.get("reliability") or {}
        base = float(reliability.get("reliability_score") or 75)
        window_30 = float((reliability.get("window_scores") or {}).get("last_30d") or base)
        window_90 = float((reliability.get("window_scores") or {}).get("last_90d") or base)

        scores: dict[str, float] = {}
        offsets = {"PRAE": 1.0, "PAAE": 2.0, "PDTS": 4.0, "SLM": -3.0, "AGE": 5.0, "CACE": 3.0}
        for module in MODULES:
            blended = window_30 * 0.4 + window_90 * 0.6
            scores[module] = round(min(100.0, max(0.0, blended + offsets.get(module, 0.0))), 2)
        return scores
