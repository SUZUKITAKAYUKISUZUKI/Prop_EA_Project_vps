"""Decision outcome evaluation from APM v1 execution log."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.apm_v2.outcome_classifier import OutcomeClassifier


class DecisionOutcomeEngine:
    EVALUATION_HORIZON_DAYS = 30

    def __init__(self, *, classifier: OutcomeClassifier | None = None) -> None:
        self._classifier = classifier or OutcomeClassifier()

    def evaluate_executed_decisions(
        self,
        decisions: list[dict[str, Any]],
        *,
        outcome_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        outcome_overrides = outcome_overrides or {}
        now = datetime.now(timezone.utc)
        outcomes: list[dict[str, Any]] = []

        for decision in decisions:
            if decision.get("status") not in {"EXECUTED", "APPROVED"}:
                continue
            decision_id = str(decision.get("action_id") or decision.get("decision_id") or "")
            if decision_id in outcome_overrides:
                item = {**decision, **outcome_overrides[decision_id]}
                outcomes.append(self._classifier.classify(self._normalize(item)))
                continue

            ts = self._parse_ts(decision.get("timestamp"))
            if ts and (now - ts).days < self.EVALUATION_HORIZON_DAYS:
                continue

            synthetic = self._synthetic_outcome(decision)
            outcomes.append(self._classifier.classify(self._normalize(synthetic)))

        return outcomes

    def _normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision_id": row.get("action_id") or row.get("decision_id"),
            "profile_id": row.get("profile_id"),
            "decision_type": row.get("action_type") or row.get("decision_type"),
            "strategy": row.get("strategy"),
            "predicted_benefit": float(row.get("predicted_benefit") or row.get("expected_benefit_pct") or 0),
            "actual_benefit": float(row.get("actual_benefit") or 0),
            "predicted_risk": float(row.get("predicted_risk") or row.get("expected_risk_pct") or 0),
            "actual_risk": float(row.get("actual_risk") or 0),
            "success_score": row.get("success_score"),
            "evaluation_date": row.get("evaluation_date") or datetime.now(timezone.utc).date().isoformat(),
            "confidence": row.get("confidence"),
        }

    def _synthetic_outcome(self, decision: dict[str, Any]) -> dict[str, Any]:
        predicted_benefit = float(decision.get("expected_benefit_pct") or decision.get("predicted_benefit") or 0)
        predicted_risk = float(decision.get("expected_risk_pct") or decision.get("predicted_risk") or 0)
        confidence = float(decision.get("confidence") or 70)
        bias = (confidence - 75.0) / 200.0
        action_type = str(decision.get("action_type") or "")
        if "PROMOTE" in action_type and confidence > 85:
            actual_benefit = predicted_benefit * (0.6 - bias)
        else:
            actual_benefit = predicted_benefit * (0.85 - bias)
        actual_risk = predicted_risk * (0.9 + bias)
        return {
            **decision,
            "predicted_benefit": predicted_benefit,
            "actual_benefit": round(actual_benefit, 4),
            "predicted_risk": predicted_risk,
            "actual_risk": round(actual_risk, 4),
        }

    def _parse_ts(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except ValueError:
            return None
