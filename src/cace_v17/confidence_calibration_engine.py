"""Confidence calibration — was stated confidence justified?"""
from __future__ import annotations

from typing import Any

from src.cace_v17.calibration_config import calibration_category


class ConfidenceCalibrationEngine:
    HIGH_CONFIDENCE_THRESHOLD = 90.0

    def evaluate(self, evaluated_records: list[dict[str, Any]]) -> dict[str, Any]:
        if not evaluated_records:
            return self._empty()

        confidences = [float(r.get("confidence") or 0) for r in evaluated_records]
        success_probs = [float(r.get("actual_success_probability") or (1.0 if r.get("actual_success") else 0.0)) for r in evaluated_records]
        mae = sum(abs(c / 100.0 - p) for c, p in zip(confidences, success_probs)) / len(confidences)
        calibration_score = round(100.0 - mae * 100.0, 2)

        high_conf = [r for r in evaluated_records if float(r.get("confidence") or 0) >= self.HIGH_CONFIDENCE_THRESHOLD]
        high_conf_calibration = self._bucket_calibration(high_conf)

        avg_confidence = sum(confidences) / len(confidences)
        avg_success_rate = sum(success_probs) / len(success_probs) * 100.0
        gap = avg_confidence - avg_success_rate

        overconfidence_score = round(max(0.0, gap), 2)
        underconfidence_score = round(max(0.0, -gap), 2)
        overconfident = gap > 10.0
        underconfident = gap < -10.0

        return {
            "calibration_score": calibration_score,
            "calibration_category": calibration_category(calibration_score),
            "mean_absolute_error": round(mae, 4),
            "high_confidence_calibration": high_conf_calibration,
            "overconfidence_score": overconfidence_score,
            "underconfidence_score": underconfidence_score,
            "overconfident": overconfident,
            "underconfident": underconfident,
            "average_confidence": round(avg_confidence, 2),
            "actual_success_rate": round(avg_success_rate, 2),
            "sample_size": len(evaluated_records),
        }

    def _bucket_calibration(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {"count": 0, "successful": 0, "calibration_pct": 0.0}
        successful = sum(1 for r in records if r.get("actual_success"))
        count = len(records)
        return {
            "count": count,
            "successful": successful,
            "calibration_pct": round(successful / count * 100.0, 2),
        }

    def _empty(self) -> dict[str, Any]:
        return {
            "calibration_score": 0.0,
            "calibration_category": "POOR",
            "mean_absolute_error": 1.0,
            "high_confidence_calibration": {"count": 0, "successful": 0, "calibration_pct": 0.0},
            "overconfidence_score": 0.0,
            "underconfidence_score": 0.0,
            "overconfident": False,
            "underconfident": False,
            "average_confidence": 0.0,
            "actual_success_rate": 0.0,
            "sample_size": 0,
        }
