"""Explainability for AI Governor decisions."""
from __future__ import annotations

import json
from typing import Any

from src.ai_governor.signals import GuardianSignal


class ExplainabilityEngine:
    """Build auditable reason_json for every governor decision."""

    def build_reason(self, signal: GuardianSignal, *, context_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "trigger": signal.reason.get("trigger") or _infer_trigger(signal),
            "decision": signal.decision_type,
            "decision_text": signal.decision,
            "recommended_action": signal.decision_type,
            "confidence": round(signal.confidence, 1),
            "source": signal.source,
            "priority": signal.priority,
            **signal.reason,
        }
        if signal.strategy_id:
            payload["strategy"] = signal.strategy_id
        if context_summary:
            payload["context"] = {
                "health_score": context_summary.get("health_score"),
                "risk_level": context_summary.get("risk_level"),
                "current_state": context_summary.get("current_state"),
                "portfolio_fit": context_summary.get("portfolio_fit"),
            }
        return payload

    def serialize(self, reason: dict[str, Any]) -> str:
        return json.dumps(reason, ensure_ascii=False, sort_keys=True)

    def deserialize(self, raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {"raw": raw}


def _infer_trigger(signal: GuardianSignal) -> str:
    if "dd" in signal.decision.lower():
        return "dd_rising"
    if signal.decision_type.endswith("STRATEGY"):
        return "strategy_lifecycle"
    if signal.decision_type == "ALLOCATION_REBALANCE":
        return "allocation_drift"
    return "governor_evaluation"
