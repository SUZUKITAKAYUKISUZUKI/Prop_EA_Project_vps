"""Component score aggregation for each Portfolio OS module."""
from __future__ import annotations

from typing import Any

from src.meta_intelligence.config import MODULES
from src.meta_intelligence.module_accuracy_engine import ModuleAccuracyEngine
from src.meta_intelligence.module_reliability_engine import ModuleReliabilityEngine
from src.meta_intelligence.module_stability_engine import ModuleStabilityEngine


class ModuleScoreEngine:
    def __init__(
        self,
        *,
        accuracy_engine: ModuleAccuracyEngine | None = None,
        stability_engine: ModuleStabilityEngine | None = None,
        reliability_engine: ModuleReliabilityEngine | None = None,
    ) -> None:
        self._accuracy = accuracy_engine or ModuleAccuracyEngine()
        self._stability = stability_engine or ModuleStabilityEngine()
        self._reliability = reliability_engine or ModuleReliabilityEngine()

    def evaluate(
        self,
        *,
        cace_v16_report: dict[str, Any] | None = None,
        cace_v17_report: dict[str, Any] | None = None,
        upstream: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, float]]:
        upstream = upstream or {}
        historical_accuracy = self._accuracy.evaluate(cace_v17_report=cace_v17_report, upstream=upstream)
        calibration = self._calibration_scores(cace_v17_report, historical_accuracy)
        stability = self._stability.evaluate(
            cace_v16_report=cace_v16_report,
            cace_v17_report=cace_v17_report,
            upstream=upstream,
        )
        consensus = self._consensus_scores(cace_v16_report)
        predictive_reliability = self._reliability.evaluate(cace_v17_report=cace_v17_report, upstream=upstream)

        components: dict[str, dict[str, float]] = {}
        for module in MODULES:
            components[module] = {
                "historical_accuracy": historical_accuracy[module],
                "calibration": calibration[module],
                "stability": stability[module],
                "consensus": consensus[module],
                "predictive_reliability": predictive_reliability[module],
            }
        return components

    def _calibration_scores(
        self,
        cace_v17_report: dict[str, Any] | None,
        accuracy_scores: dict[str, float],
    ) -> dict[str, float]:
        report = cace_v17_report or {}
        base_calibration = float(report.get("calibration_score") or 75)
        evaluated = report.get("evaluated_decisions") or []
        module_rows: dict[str, list[float]] = {m: [] for m in MODULES}

        for row in evaluated:
            module = str(row.get("module") or "CACE").upper()
            if module not in module_rows:
                continue
            conf = float(row.get("confidence") or 0)
            success = 1.0 if row.get("actual_success") else 0.0
            module_rows[module].append(100.0 - abs(conf / 100.0 - success) * 100.0)

        scores: dict[str, float] = {}
        offsets = {"PRAE": -1.0, "PAAE": 0.5, "PDTS": -3.0, "SLM": -2.0, "AGE": 2.0, "CACE": 1.5}
        for module in MODULES:
            rows = module_rows.get(module) or []
            if rows:
                scores[module] = round(sum(rows) / len(rows), 2)
            else:
                acc = accuracy_scores.get(module, base_calibration)
                scores[module] = round(
                    min(100.0, max(0.0, base_calibration * 0.6 + acc * 0.4 + offsets.get(module, 0))),
                    2,
                )
        return scores

    def _consensus_scores(self, cace_v16_report: dict[str, Any] | None) -> dict[str, float]:
        v16 = cace_v16_report or {}
        agreements = (v16.get("consensus") or {}).get("module_agreements") or v16.get("module_agreements") or {}
        base_consensus = float(v16.get("consensus_score") or 75)
        scores: dict[str, float] = {}
        recommended = str(v16.get("recommended_action") or "")
        for module in MODULES:
            entry = agreements.get(module) or {}
            if entry:
                conf = float(entry.get("confidence") or base_consensus)
                action = str(entry.get("action") or "")
                aligned = 100.0 if action == recommended else 70.0
                scores[module] = round((conf + aligned) / 2.0, 2)
            else:
                scores[module] = round(base_consensus, 2)
        return scores
