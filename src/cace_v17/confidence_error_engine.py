"""Prediction error and confidence accuracy calculations."""
from __future__ import annotations

from typing import Any

from src.cace_v17.calibration_config import SUCCESS_BENEFIT_THRESHOLD, SUCCESS_DD_TOLERANCE


class ConfidenceErrorEngine:
    def evaluate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        predicted_benefit = float(record.get("predicted_benefit") or 0)
        actual_benefit = float(record.get("actual_benefit") or 0)
        predicted_dd = float(record.get("predicted_dd") or 0)
        actual_dd = float(record.get("actual_dd") or 0)
        confidence = float(record.get("confidence") or 0)

        benefit_error = actual_benefit - predicted_benefit
        dd_error = actual_dd - predicted_dd
        prediction_error = round((abs(benefit_error) + abs(dd_error)) / 2.0, 4)

        success = self._is_successful(record)
        actual_success_probability = 1.0 if success else 0.0
        confidence_accuracy = round(100.0 - abs(confidence - actual_success_probability * 100.0), 2)
        outcome_accuracy = round(100.0 if success else 0.0, 2)

        return {
            **record,
            "benefit_error": round(benefit_error, 4),
            "dd_error": round(dd_error, 4),
            "prediction_error": prediction_error,
            "confidence_accuracy": confidence_accuracy,
            "outcome_accuracy": outcome_accuracy,
            "actual_success": success,
            "actual_success_probability": actual_success_probability,
        }

    def evaluate_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.evaluate_record(record) for record in records]

    def _is_successful(self, record: dict[str, Any]) -> bool:
        if record.get("actual_success") is not None:
            return bool(record.get("actual_success"))
        actual_benefit = float(record.get("actual_benefit") or 0)
        predicted_benefit = float(record.get("predicted_benefit") or 0)
        actual_dd = float(record.get("actual_dd") or 0)
        predicted_dd = float(record.get("predicted_dd") or 0)
        benefit_ok = actual_benefit >= max(SUCCESS_BENEFIT_THRESHOLD, predicted_benefit * 0.5)
        dd_ok = actual_dd >= predicted_dd - SUCCESS_DD_TOLERANCE
        return benefit_ok and dd_ok
