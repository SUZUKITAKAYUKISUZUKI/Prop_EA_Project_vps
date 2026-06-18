"""Historical accuracy scoring per Portfolio OS module."""
from __future__ import annotations

from typing import Any

from src.meta_intelligence.config import MODULES


class ModuleAccuracyEngine:
    def evaluate(
        self,
        *,
        cace_v17_report: dict[str, Any] | None = None,
        upstream: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        upstream = upstream or {}
        rec_scores = (cace_v17_report or {}).get("recommendation_accuracy") or {}
        scores: dict[str, float] = {}

        for module in MODULES:
            if module in rec_scores and float(rec_scores[module] or 0) > 0:
                scores[module] = round(float(rec_scores[module]), 2)
                continue
            scores[module] = round(self._infer_accuracy(module, upstream, cace_v17_report), 2)
        return scores

    def _infer_accuracy(
        self,
        module: str,
        upstream: dict[str, Any],
        cace_v17_report: dict[str, Any] | None,
    ) -> float:
        if module == "PRAE":
            health = float((upstream.get("prae_v2") or {}).get("health_report", {}).get("health_score") or 78)
            return min(100.0, health + 4.0)
        if module == "PAAE":
            quality = (upstream.get("paae") or {}).get("quality_scores") or {}
            return min(100.0, sum(float(v) for v in quality.values()) / max(len(quality), 1))
        if module == "PDTS":
            ranking = (upstream.get("pdts") or {}).get("recommendation_ranking") or []
            if ranking:
                return float(ranking[0].get("score") or ranking[0].get("pass_rate") or 82)
            return 82.0
        if module == "SLM":
            strategies = (upstream.get("slm") or {}).get("strategies") or []
            if strategies:
                fits = [float(s.get("portfolio_fit_score") or 70) for s in strategies]
                return sum(fits) / len(fits)
            return 76.0
        if module == "AGE":
            age = upstream.get("age_v4") or {}
            return float(age.get("strategic_score") or age.get("strategic_confidence") or 84)
        if module == "CACE":
            report = cace_v17_report or {}
            return float(report.get("calibration_score") or report.get("confidence") or 88)
        return 70.0
