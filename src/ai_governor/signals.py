"""Shared guardian signal types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_governor.decision_types import DecisionType


@dataclass
class GuardianSignal:
    decision_type: str
    decision: str
    confidence: float
    reason: dict[str, Any] = field(default_factory=dict)
    source: str = "guardian"
    strategy_id: str | None = None
    priority: str = "MEDIUM"
    expected_benefit: float = 0.0
    expected_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "strategy_id": self.strategy_id,
            "priority": self.priority,
            "expected_benefit": self.expected_benefit,
            "expected_risk": self.expected_risk,
        }

    @property
    def action(self) -> str:
        return self.decision_type
