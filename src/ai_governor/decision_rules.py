"""Configurable thresholds for AI Governor Engine."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GovernorDecisionRules:
    """All AGE thresholds — override per deployment without code changes."""

    # Risk guardian
    dd_budget_pct: float = 100.0
    dd_recommend_recovery_pct: float = 70.0
    dd_force_recovery_pct: float = 85.0
    risk_score_alert_threshold: float = 70.0

    # Allocation guardian
    rebalance_score_gap_pct: float = 10.0
    rebalance_ignore_gap_pct: float = 5.0

    # Strategy guardian (SLM v3)
    promote_score_threshold: float = 75.0
    promote_fit_threshold: float = 60.0
    core_promote_score_threshold: float = 85.0
    core_promote_fit_threshold: float = 80.0
    demote_fit_threshold: float = 40.0
    retire_fit_threshold: float = 20.0
    retire_score_threshold: float = 50.0

    # Health monitor
    health_excellent: float = 90.0
    health_good: float = 75.0
    health_fair: float = 60.0
    health_at_risk: float = 40.0

    # Confidence
    high_confidence: float = 95.0
    medium_confidence: float = 80.0
    low_confidence: float = 60.0

    # Agreement weights for confidence scoring
    system_weights: dict[str, float] = field(
        default_factory=lambda: {
            "paae": 0.20,
            "pdts": 0.20,
            "slm": 0.20,
            "prae": 0.25,
            "state": 0.15,
        }
    )


DEFAULT_RULES = GovernorDecisionRules()
