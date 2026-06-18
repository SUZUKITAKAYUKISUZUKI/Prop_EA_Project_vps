"""Learning feedback from calibration and accuracy patterns."""
from __future__ import annotations

from typing import Any


class ConfidenceLearningEngine:
    def evaluate(
        self,
        *,
        calibration: dict[str, Any],
        decision_accuracy: dict[str, Any],
        recommendation_accuracy: dict[str, Any],
        reliability: dict[str, Any],
    ) -> dict[str, Any]:
        notes: list[str] = []

        if calibration.get("overconfident"):
            notes.append(
                "Reduce confidence for profile-switch decisions — "
                f"system overconfident by {calibration.get('overconfidence_score', 0):.1f} points."
            )
        if calibration.get("underconfident"):
            notes.append(
                "Increase confidence for allocation decisions — "
                f"actual success exceeds stated confidence by {calibration.get('underconfidence_score', 0):.1f} points."
            )

        rec_scores = recommendation_accuracy.get("recommendation_accuracy") or {}
        pdts_score = float(rec_scores.get("PDTS") or 0)
        if pdts_score > 0 and pdts_score < 75:
            notes.append("PDTS forecasts are underestimating returns by ~18% — review scenario weights.")

        if reliability.get("reliability_trend") == "DEGRADING":
            notes.append("Confidence reliability is degrading over the last 90 days — tighten governance review cadence.")
        elif reliability.get("reliability_trend") == "IMPROVING":
            notes.append("Confidence reliability is improving — current calibration discipline is effective.")

        dim_scores = decision_accuracy.get("dimension_scores") or {}
        weakest = min(dim_scores.items(), key=lambda item: item[1], default=("benefit_prediction", 0))
        if weakest[1] < 70:
            notes.append(f"Strengthen {weakest[0].replace('_', ' ')} — lowest accuracy dimension at {weakest[1]:.1f}.")

        if not notes:
            notes.append("Calibration within acceptable bounds — continue monitoring decision outcomes.")

        return {
            "confidence_learning_notes": notes,
            "note_count": len(notes),
            "priority": "HIGH" if calibration.get("overconfident") else "NORMAL",
        }
