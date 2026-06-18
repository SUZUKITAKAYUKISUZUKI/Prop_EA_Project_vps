"""Dashboard report builder for AI Governor."""
from __future__ import annotations

from typing import Any

from src.ai_governor.allocation_guardian import AllocationAssessment
from src.ai_governor.decision_engine import GovernorDecision
from src.ai_governor.decision_types import DecisionType
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.health_monitor import PortfolioHealthSnapshot
from src.ai_governor.profile_guardian import ProfileAssessment
from src.ai_governor.recommendation_engine import GovernorRecommendation
from src.ai_governor.risk_guardian import RiskAssessment
from src.ai_governor.state_guardian import StateAssessment
from src.ai_governor.strategy_guardian import StrategyAssessment


class GovernorReport:
    def build(
        self,
        *,
        context: GovernorContext,
        health: PortfolioHealthSnapshot,
        assessments: dict[str, Any],
        decisions: list[GovernorDecision],
        recommendations: list[GovernorRecommendation],
        history: list[dict[str, Any]] | None = None,
        health_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        risk: RiskAssessment = assessments["risk"]
        state: StateAssessment = assessments["state"]
        profile: ProfileAssessment = assessments["profile"]

        alert_types = {
            DecisionType.RISK_ALERT.value,
            DecisionType.HEALTH_ALERT.value,
            DecisionType.ENTER_RECOVERY.value,
            DecisionType.REDUCE_RISK.value,
        }
        alerts = [d for d in decisions if d.decision_type in alert_types]
        opportunities = [
            r.to_dict()
            for r in recommendations
            if r.category in {"INFO", "LOW"} or r.action == DecisionType.PROMOTE_STRATEGY.value
        ]
        risks = [
            d.to_dict()
            for d in decisions
            if d.decision_type
            in {
                DecisionType.RISK_ALERT.value,
                DecisionType.HEALTH_ALERT.value,
                DecisionType.DEMOTE_STRATEGY.value,
                DecisionType.RETIRE_STRATEGY.value,
                DecisionType.ENTER_RECOVERY.value,
                DecisionType.REDUCE_RISK.value,
            }
        ]
        actions = [
            r.to_dict()
            for r in recommendations
            if r.category in {"CRITICAL", "HIGH", "MEDIUM"} or r.priority in {"CRITICAL", "HIGH", "MEDIUM"}
        ]
        avg_confidence = round(sum(d.confidence for d in decisions) / len(decisions), 1) if decisions else 0.0

        return {
            "profile_id": context.profile_id,
            "profile_name": context.profile_name,
            "current_state": context.current_state,
            "current_profile": context.current_profile,
            "objective_label": context.objective_label,
            "health": health.to_dict(),
            "health_score": health.health_score,
            "health_status": health.health_status,
            "risk_level": health.risk_level,
            "risk_assessment": risk.to_dict(),
            "state_assessment": state.to_dict(),
            "profile_assessment": profile.to_dict(),
            "allocation_assessment": assessments["allocation"].to_dict(),
            "strategy_assessment": assessments["strategy"].to_dict(),
            "portfolio_fit": context.portfolio_fit,
            "pass_rate": context.pass_rate,
            "dd_pct": context.dd_pct,
            "confidence": avg_confidence,
            "open_alerts": [a.to_dict() for a in alerts],
            "recommended_actions": actions,
            "recent_decisions": [d.to_dict() for d in decisions[:10]],
            "decision_timeline": history or [],
            "decision_history": history or [],
            "top_risks": risks[:5],
            "top_opportunities": opportunities[:5],
            "recommendations": [r.to_dict() for r in recommendations],
            "context_summary": context.to_dict(),
            "health_history": health_history or [],
        }

    def status_payload(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            "profile_id": report.get("profile_id"),
            "current_state": report.get("current_state"),
            "current_profile": report.get("current_profile"),
            "health_score": report.get("health_score"),
            "health_status": report.get("health_status"),
            "risk_level": report.get("risk_level"),
            "confidence": report.get("confidence"),
            "open_alert_count": len(report.get("open_alerts") or []),
            "recommended_action_count": len(report.get("recommended_actions") or []),
        }
