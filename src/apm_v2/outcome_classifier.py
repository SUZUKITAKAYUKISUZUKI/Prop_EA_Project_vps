"""Outcome classification for executive decisions."""
from __future__ import annotations

from typing import Any

from src.apm_v2.config import OUTCOME_SUCCESS_THRESHOLD


class OutcomeClassifier:
    def classify(self, outcome: dict[str, Any]) -> dict[str, Any]:
        predicted_benefit = float(outcome.get("predicted_benefit") or 0)
        actual_benefit = float(outcome.get("actual_benefit") or 0)
        predicted_risk = float(outcome.get("predicted_risk") or 0)
        actual_risk = float(outcome.get("actual_risk") or 0)
        success_score = float(outcome.get("success_score") or self._success_score(outcome))

        predicted_benefit = float(outcome.get("predicted_benefit") or 0)
        actual_benefit = float(outcome.get("actual_benefit") or 0)
        if predicted_benefit > 0 and actual_benefit < 0:
            result = "FAILURE"
            success_score = min(success_score, 35.0)
        elif success_score >= OUTCOME_SUCCESS_THRESHOLD:
            result = "SUCCESS"
        elif success_score >= 40:
            result = "PARTIAL"
        else:
            result = "FAILURE"

        return {
            **outcome,
            "success_score": round(success_score, 2),
            "outcome_class": result,
            "benefit_delta": round(actual_benefit - predicted_benefit, 4),
            "risk_delta": round(actual_risk - predicted_risk, 4),
        }

    def _success_score(self, outcome: dict[str, Any]) -> float:
        predicted_benefit = float(outcome.get("predicted_benefit") or 0)
        actual_benefit = float(outcome.get("actual_benefit") or 0)
        predicted_risk = float(outcome.get("predicted_risk") or 0)
        actual_risk = float(outcome.get("actual_risk") or 0)

        benefit_score = 100.0 - min(100.0, abs(actual_benefit - predicted_benefit) * 15.0)
        risk_score = 100.0 - min(100.0, abs(actual_risk - predicted_risk) * 15.0)
        direction_bonus = 10.0 if actual_benefit >= predicted_benefit * 0.5 else 0.0
        return max(0.0, min(100.0, benefit_score * 0.6 + risk_score * 0.4 + direction_bonus))
