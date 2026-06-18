"""Executive context aggregation for APM v1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutiveContext:
    profile_id: str
    portfolio_state: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    allocation: dict[str, Any] = field(default_factory=dict)
    strategy_lifecycle: dict[str, Any] = field(default_factory=dict)
    governor_decisions: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    trust_scores: dict[str, Any] = field(default_factory=dict)
    upstream: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "portfolio_state": self.portfolio_state,
            "profile": self.profile,
            "health": self.health,
            "risk": self.risk,
            "allocation": self.allocation,
            "strategy_lifecycle": self.strategy_lifecycle,
            "governor_decisions": self.governor_decisions,
            "confidence": self.confidence,
            "trust_scores": self.trust_scores,
        }


class ExecutiveContextBuilder:
    def build(
        self,
        *,
        profile_id: str,
        upstream: dict[str, Any],
        cace_v16_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        mie_report: dict[str, Any],
    ) -> ExecutiveContext:
        prae = upstream.get("prae_v2") or {}
        paae = upstream.get("paae") or {}
        slm = upstream.get("slm") or {}
        age_v4 = upstream.get("age_v4") or {}
        state = upstream.get("state_analytics") or {}

        health_report = prae.get("health_report") or {}
        current_weights = paae.get("current_weights") or paae.get("strategy_allocations") or {}
        recommended = paae.get("recommended_weights") or {}

        return ExecutiveContext(
            profile_id=profile_id,
            portfolio_state={
                "current_state": state.get("current_state"),
                "health_status": health_report.get("health_status"),
                "recovery_events": health_report.get("recovery_events", 0),
            },
            profile={
                "profile_id": profile_id,
                "profile_type": state.get("profile_type") or "funded",
                "account_state": state.get("current_state"),
            },
            health=health_report,
            risk={
                "strategy_risk": prae.get("strategy_risk") or [],
                "dd_attribution": prae.get("dd_attribution") or {},
                "highest_risk_strategy": self._highest_risk(prae),
            },
            allocation={
                "current_weights": current_weights,
                "recommended_weights": recommended,
                "drift_alerts": paae.get("drift_alerts") or [],
            },
            strategy_lifecycle={
                "strategies": slm.get("strategies") or [],
                "promotion_candidates": slm.get("promotion_candidates") or [],
                "retirement_candidates": slm.get("retirement_candidates") or [],
            },
            governor_decisions={
                "recommended_action": age_v4.get("recommended_action"),
                "strategic_score": age_v4.get("strategic_score"),
                "strategic_confidence": age_v4.get("strategic_confidence"),
                "rank_category": age_v4.get("rank_category"),
            },
            confidence={
                "portfolio_confidence": cace_v17_report.get("confidence"),
                "calibration_score": cace_v17_report.get("calibration_score"),
                "consensus_score": cace_v16_report.get("consensus_score"),
                "durability_score": cace_v16_report.get("durability_score"),
                "recommended_action": cace_v16_report.get("recommended_action"),
            },
            trust_scores=mie_report.get("module_trust_scores") or {},
            upstream=upstream,
        )

    def _highest_risk(self, prae: dict[str, Any]) -> str | None:
        risks = prae.get("strategy_risk") or []
        if not risks:
            return None
        top = max(risks, key=lambda r: float(r.get("risk_score") or 0))
        return str(top.get("strategy") or "")
