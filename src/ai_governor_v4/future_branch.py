"""Future branch model for AGE v4 multi-future simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FutureBranch:
    branch_id: str
    action_type: str
    action_label: str
    description: str
    metrics_by_horizon: dict[str, dict[str, float]] = field(default_factory=dict)
    strategic_score: float = 0.0
    rank_category: str = "REJECT"
    confidence: float = 0.0
    expected_benefit: float = 0.0
    expected_risk: float = 0.0
    modification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "action_type": self.action_type,
            "action_label": self.action_label,
            "description": self.description,
            "metrics_by_horizon": self.metrics_by_horizon,
            "strategic_score": round(self.strategic_score, 1),
            "rank_category": self.rank_category,
            "confidence": round(self.confidence, 1),
            "expected_benefit": round(self.expected_benefit, 2),
            "expected_risk": round(self.expected_risk, 2),
            "modification": self.modification,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FutureBranch:
        return cls(
            branch_id=str(data.get("branch_id") or ""),
            action_type=str(data.get("action_type") or ""),
            action_label=str(data.get("action_label") or ""),
            description=str(data.get("description") or ""),
            metrics_by_horizon=dict(data.get("metrics_by_horizon") or {}),
            strategic_score=float(data.get("strategic_score") or 0),
            rank_category=str(data.get("rank_category") or "REJECT"),
            confidence=float(data.get("confidence") or 0),
            expected_benefit=float(data.get("expected_benefit") or 0),
            expected_risk=float(data.get("expected_risk") or 0),
            modification=dict(data.get("modification") or {}),
        )
