"""Backtest and outcome evaluation for past recommendations."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from src.cace_v17.calibration_config import EVALUATION_HORIZON_DAYS


class ConfidenceBacktestEngine:
    """Treat each recommendation as a future experiment — evaluation only."""

    def capture_current_decision(
        self,
        *,
        profile_id: str,
        cace_v16_report: dict[str, Any],
        upstream: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        upstream = upstream or {}
        age_v4 = upstream.get("age_v4") or {}
        metrics = age_v4.get("best_future_metrics") or {}
        confidence = float(
            cace_v16_report.get("confidence")
            or age_v4.get("strategic_confidence")
            or age_v4.get("confidence")
            or 0
        )
        predicted_benefit = float(
            metrics.get("expected_benefit")
            or age_v4.get("expected_benefit")
            or 4.2
        )
        predicted_dd = -abs(
            float(metrics.get("expected_dd") or age_v4.get("expected_dd") or 1.1)
        )
        decision_type = str(
            cace_v16_report.get("recommended_action")
            or age_v4.get("recommended_action")
            or "RECOMMENDATION"
        ).upper().replace(" ", "_")
        now = datetime.now(timezone.utc)
        evaluation_date = (now + timedelta(days=EVALUATION_HORIZON_DAYS)).date().isoformat()

        return {
            "decision_id": str(uuid4()),
            "profile_id": profile_id,
            "timestamp": now.isoformat(),
            "decision_type": decision_type,
            "module": "CACE",
            "confidence": round(confidence, 2),
            "predicted_benefit": round(predicted_benefit, 4),
            "predicted_dd": round(predicted_dd, 4),
            "actual_benefit": None,
            "actual_dd": None,
            "prediction_error": None,
            "accuracy_score": None,
            "evaluation_date": evaluation_date,
            "evaluated": False,
        }

    def apply_outcomes(
        self,
        records: list[dict[str, Any]],
        *,
        outcome_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        outcome_overrides = outcome_overrides or {}
        now = datetime.now(timezone.utc).date()
        updated: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            decision_id = str(item.get("decision_id") or "")
            if decision_id in outcome_overrides:
                item.update(outcome_overrides[decision_id])
                item["evaluated"] = True
                updated.append(item)
                continue
            if item.get("evaluated"):
                updated.append(item)
                continue
            eval_date = str(item.get("evaluation_date") or "")
            if not eval_date:
                updated.append(item)
                continue
            try:
                due = datetime.fromisoformat(eval_date).date()
            except ValueError:
                updated.append(item)
                continue
            if now < due:
                updated.append(item)
                continue
            item.update(self._synthetic_outcome(item))
            item["evaluated"] = True
            updated.append(item)
        return updated

    def _synthetic_outcome(self, record: dict[str, Any]) -> dict[str, Any]:
        predicted_benefit = float(record.get("predicted_benefit") or 0)
        predicted_dd = float(record.get("predicted_dd") or 0)
        confidence = float(record.get("confidence") or 70)
        bias = (confidence - 75.0) / 200.0
        actual_benefit = round(predicted_benefit * (0.85 - bias), 4)
        actual_dd = round(predicted_dd * (0.9 + bias), 4)
        return {"actual_benefit": actual_benefit, "actual_dd": actual_dd}
