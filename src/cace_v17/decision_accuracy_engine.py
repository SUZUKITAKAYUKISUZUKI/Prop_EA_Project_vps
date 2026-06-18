"""Decision accuracy scoring across prediction dimensions."""
from __future__ import annotations

from typing import Any


class DecisionAccuracyEngine:
    DIMENSIONS = (
        "benefit_prediction",
        "dd_prediction",
        "recovery_prediction",
        "profile_prediction",
        "allocation_prediction",
    )

    def evaluate(self, evaluated_records: list[dict[str, Any]]) -> dict[str, Any]:
        if not evaluated_records:
            return self._empty()

        dimension_scores = {dim: [] for dim in self.DIMENSIONS}
        for record in evaluated_records:
            scores = self._score_record(record)
            for dim, value in scores.items():
                dimension_scores[dim].append(value)

        breakdown = {
            dim: round(sum(values) / len(values), 2) if values else 0.0
            for dim, values in dimension_scores.items()
        }
        active = [v for v in breakdown.values() if v > 0]
        decision_accuracy_score = round(sum(active) / len(active), 2) if active else 0.0

        return {
            "decision_accuracy_score": decision_accuracy_score,
            "dimension_scores": breakdown,
            "evaluated_count": len(evaluated_records),
        }

    def _score_record(self, record: dict[str, Any]) -> dict[str, float]:
        predicted_benefit = float(record.get("predicted_benefit") or 0)
        actual_benefit = float(record.get("actual_benefit") or 0)
        predicted_dd = float(record.get("predicted_dd") or 0)
        actual_dd = float(record.get("actual_dd") or 0)
        confidence_accuracy = float(record.get("confidence_accuracy") or record.get("accuracy_score") or 0)

        benefit_score = self._closeness_score(predicted_benefit, actual_benefit, scale=5.0)
        dd_score = self._closeness_score(predicted_dd, actual_dd, scale=3.0)
        recovery_score = benefit_score * 0.9 if "RECOVERY" in str(record.get("decision_type", "")).upper() else benefit_score
        profile_score = confidence_accuracy if "PROFILE" in str(record.get("decision_type", "")).upper() else benefit_score
        allocation_score = benefit_score if "ALLOCATION" in str(record.get("decision_type", "")).upper() or "REBALANCE" in str(record.get("decision_type", "")).upper() else benefit_score

        return {
            "benefit_prediction": benefit_score,
            "dd_prediction": dd_score,
            "recovery_prediction": recovery_score,
            "profile_prediction": profile_score,
            "allocation_prediction": allocation_score,
        }

    def _closeness_score(self, predicted: float, actual: float, *, scale: float) -> float:
        error = abs(actual - predicted)
        return round(max(0.0, 100.0 - (error / max(scale, 0.01)) * 100.0), 2)

    def _empty(self) -> dict[str, Any]:
        return {
            "decision_accuracy_score": 0.0,
            "dimension_scores": {dim: 0.0 for dim in self.DIMENSIONS},
            "evaluated_count": 0,
        }
