"""Combined MIE v1 meta intelligence report."""
from __future__ import annotations

from typing import Any


class MetaIntelligenceReport:
    def build(
        self,
        *,
        profile_id: str,
        trust_scores: dict[str, dict[str, Any]],
        rankings: list[dict[str, Any]],
        drift: dict[str, Any],
        recommendations: dict[str, Any],
        improvements: list[dict[str, Any]],
        strongest_weakest: dict[str, Any],
        cace_v17_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            "module_trust_scores": trust_scores,
            "module_rankings": rankings,
            "strongest_module": strongest_weakest.get("strongest_module"),
            "weakest_module": strongest_weakest.get("weakest_module"),
            "drift": drift,
            "drift_alerts": drift.get("drift_alerts"),
            "recommendations": recommendations,
            "self_improvement_notes": improvements,
            "calibration_score": (cace_v17_report or {}).get("calibration_score"),
            "decision_accuracy_score": (cace_v17_report or {}).get("decision_accuracy_score"),
        }
