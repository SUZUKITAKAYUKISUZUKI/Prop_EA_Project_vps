"""Combined CACE v1.7 calibration intelligence report."""
from __future__ import annotations

from typing import Any


class CalibrationReport:
    def build(
        self,
        *,
        profile_id: str,
        cace_v16_report: dict[str, Any],
        calibration: dict[str, Any],
        decision_accuracy: dict[str, Any],
        recommendation_accuracy: dict[str, Any],
        reliability: dict[str, Any],
        learning: dict[str, Any],
        evaluated_records: list[dict[str, Any]],
        pending_decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            "confidence": cace_v16_report.get("confidence"),
            "consensus_score": cace_v16_report.get("consensus_score"),
            "durability_score": cace_v16_report.get("durability_score"),
            "recommended_action": cace_v16_report.get("recommended_action"),
            "calibration_score": calibration.get("calibration_score"),
            "calibration_category": calibration.get("calibration_category"),
            "overconfidence_score": calibration.get("overconfidence_score"),
            "underconfidence_score": calibration.get("underconfidence_score"),
            "overconfident": calibration.get("overconfident"),
            "underconfident": calibration.get("underconfident"),
            "high_confidence_calibration": calibration.get("high_confidence_calibration"),
            "decision_accuracy_score": decision_accuracy.get("decision_accuracy_score"),
            "decision_accuracy": decision_accuracy,
            "recommendation_accuracy": recommendation_accuracy.get("recommendation_accuracy"),
            "recommendation_accuracy_detail": recommendation_accuracy,
            "reliability": reliability,
            "reliability_trend": reliability.get("reliability_trend"),
            "reliability_score": reliability.get("reliability_score"),
            "confidence_learning_notes": learning.get("confidence_learning_notes"),
            "learning": learning,
            "calibration": calibration,
            "evaluated_decisions": evaluated_records,
            "pending_decisions": pending_decisions,
            "cace_v16": cace_v16_report,
        }
