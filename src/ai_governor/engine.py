"""AI Governor Engine orchestrator."""
from __future__ import annotations

from typing import Any

from src.ai_governor.allocation_guardian import AllocationAssessment, AllocationGuardian
from src.ai_governor.decision_engine import DecisionEngine
from src.ai_governor.decision_history import DecisionHistory
from src.ai_governor.decision_repository import DecisionRepository
from src.ai_governor.governor_context import GovernorContext
from src.ai_governor.governor_report import GovernorReport
from src.ai_governor.health_monitor import HealthMonitor
from src.ai_governor.profile_guardian import ProfileAssessment, ProfileGuardian
from src.ai_governor.recommendation_engine import RecommendationEngine
from src.ai_governor.risk_guardian import RiskAssessment, RiskGuardian
from src.ai_governor.signals import GuardianSignal
from src.ai_governor.state_guardian import StateAssessment, StateGuardian
from src.ai_governor.strategy_guardian import StrategyAssessment, StrategyGuardian


class AiGovernorEngine:
    """Top-level portfolio governance — recommends actions, never executes trades or MT5 changes."""

    def __init__(
        self,
        *,
        repo: DecisionRepository | None = None,
        history: DecisionHistory | None = None,
        health_monitor: HealthMonitor | None = None,
        risk_guardian: RiskGuardian | None = None,
        state_guardian: StateGuardian | None = None,
        profile_guardian: ProfileGuardian | None = None,
        allocation_guardian: AllocationGuardian | None = None,
        strategy_guardian: StrategyGuardian | None = None,
        decision_engine: DecisionEngine | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        reporter: GovernorReport | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections or repo is None
        self._repo = repo or DecisionRepository(owns_connection=self._owns)
        self._history = history or DecisionHistory(self._repo, owns_connection=False)
        self._health = health_monitor or HealthMonitor()
        self._risk = risk_guardian or RiskGuardian()
        self._state = state_guardian or StateGuardian()
        self._profile = profile_guardian or ProfileGuardian()
        self._allocation = allocation_guardian or AllocationGuardian()
        self._strategy = strategy_guardian or StrategyGuardian()
        self._decisions = decision_engine or DecisionEngine()
        self._recommendations = recommendation_engine or RecommendationEngine()
        self._reporter = reporter or GovernorReport()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()
            self._history.close()

    def load_context(
        self,
        *,
        profile_ctx: dict[str, Any],
        prae_v2: dict[str, Any],
        state_analytics: dict[str, Any],
        paae: dict[str, Any],
        pdts: dict[str, Any],
        slm: dict[str, Any],
    ) -> GovernorContext:
        return GovernorContext.from_payload(
            profile_ctx=profile_ctx,
            prae_v2=prae_v2,
            state_analytics=state_analytics,
            paae=paae,
            pdts=pdts,
            slm=slm,
        )

    def run_assessments(self, context: GovernorContext) -> dict[str, Any]:
        return {
            "risk": self._risk.assess(context),
            "state": self._state.assess(context),
            "profile": self._profile.assess(context),
            "allocation": self._allocation.assess(context),
            "strategy": self._strategy.assess(context),
        }

    def evaluate_guardians(self, context: GovernorContext) -> list[GuardianSignal]:
        assessments = self.run_assessments(context)
        signals: list[GuardianSignal] = []
        for assessment in assessments.values():
            signals.extend(assessment.signals)
        return signals

    def run_governor_cycle(
        self,
        context: GovernorContext,
        *,
        persist: bool = True,
        created_by: str = "age_weekly_cycle",
    ) -> dict[str, Any]:
        health = self._health.evaluate(context)
        assessments = self.run_assessments(context)
        signals = self.evaluate_guardians(context)
        decisions = self._decisions.evaluate(context, signals)
        recommendations = self._recommendations.build(context, decisions)

        if persist:
            self._repo.save_decisions(decisions, created_by=created_by)
            self._repo.save_recommendations(recommendations)
            self._repo.save_health_snapshot(health, profile_id=context.profile_id)

        timeline = self._history.timeline(profile_id=context.profile_id, limit=20)
        health_history = self._history.health_history(limit=10)
        report = self._reporter.build(
            context=context,
            health=health,
            assessments=assessments,
            decisions=decisions,
            recommendations=recommendations,
            history=timeline,
            health_history=health_history,
        )
        self._last_report = report
        return report

    def get_governor_status(self) -> dict[str, Any]:
        if self._last_report:
            return self._reporter.status_payload(self._last_report)
        latest_health = self._repo.list_health_snapshots(limit=1)
        if latest_health:
            row = latest_health[0]
            return {
                "health_score": row.get("health_score"),
                "health_status": row.get("health_status"),
                "risk_level": row.get("risk_level"),
                "current_state": row.get("state"),
                "current_profile": row.get("profile") or row.get("profile_id"),
                "confidence": 0.0,
                "open_alert_count": len(self._repo.list_recommendations(status="OPEN", limit=100)),
            }
        return {"health_score": None, "health_status": "UNKNOWN", "risk_level": "UNKNOWN"}

    def get_governor_report(self, report: dict[str, Any] | None = None) -> dict[str, Any]:
        return report or self._last_report or {}

    def get_governor_recommendations(self, *, status: str = "OPEN") -> list[dict[str, Any]]:
        return self._repo.list_recommendations(status=status, limit=100)

    def get_decision_history(
        self,
        *,
        profile_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._history.timeline(profile_id=profile_id, limit=limit)
