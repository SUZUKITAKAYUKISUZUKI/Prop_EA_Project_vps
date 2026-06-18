"""Module output stability scoring."""
from __future__ import annotations

from typing import Any

from src.meta_intelligence.config import MODULES


class ModuleStabilityEngine:
    def evaluate(
        self,
        *,
        cace_v16_report: dict[str, Any] | None = None,
        cace_v17_report: dict[str, Any] | None = None,
        upstream: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        upstream = upstream or {}
        v16 = cace_v16_report or {}
        v17 = cace_v17_report or {}
        cace_stability = float(
            (v16.get("cace_v15") or {}).get("confidence_stability")
            or v16.get("durability_score")
            or 80
        )

        scores: dict[str, float] = {}
        for module in MODULES:
            scores[module] = round(self._module_stability(module, upstream, cace_stability, v17), 2)
        return scores

    def _module_stability(
        self,
        module: str,
        upstream: dict[str, Any],
        cace_stability: float,
        v17: dict[str, Any],
    ) -> float:
        if module == "PRAE":
            return float((upstream.get("prae_v2") or {}).get("health_report", {}).get("health_score") or 78)
        if module == "PAAE":
            current = (upstream.get("paae") or {}).get("current_weights") or {}
            recommended = (upstream.get("paae") or {}).get("recommended_weights") or {}
            if not current:
                return 80.0
            drift = sum(abs(float(recommended.get(k, 0)) - float(current.get(k, 0))) for k in current)
            return max(40.0, 100.0 - drift * 100.0)
        if module == "PDTS":
            scenarios = (upstream.get("pdts") or {}).get("scenario_comparison") or {}
            return float(scenarios.get("recommended", {}).get("score") or 82)
        if module == "SLM":
            strategies = (upstream.get("slm") or {}).get("strategies") or []
            stable = sum(1 for s in strategies if str(s.get("stage", "")).upper() in {"MATURE", "GROWTH"})
            return min(100.0, 60.0 + stable * 8.0)
        if module == "AGE":
            age = upstream.get("age_v4") or {}
            return float(age.get("strategic_confidence") or age.get("strategic_score") or 84)
        if module == "CACE":
            return cace_stability
        return 70.0
