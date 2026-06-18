"""APM v1 main orchestration — executive layer, no broker execution."""
from __future__ import annotations

from typing import Any

from src.apm.config import (
    CACHE_APM_OPPORTUNITIES,
    CACHE_APM_QUEUE,
    CACHE_APM_RISK_ALERTS,
    CACHE_APM_ROADMAP,
    CACHE_APM_STATUS,
)
from src.apm.executive_context import ExecutiveContextBuilder
from src.apm.executive_engine import ExecutiveEngine
from src.apm.executive_planner import ExecutivePlanner
from src.apm.executive_report import ExecutiveReport
from src.apm.executive_repository import ExecutiveRepository
from src.apm.execution_queue import ExecutionQueue
from src.apm.governance_engine import GovernanceEngine
from src.apm.opportunity_engine import OpportunityEngine
from src.apm.policy_engine import PolicyEngine
from src.apm.recommendation_engine import RecommendationEngine
from src.apm.risk_engine import RiskEngine
from src.apm.roadmap_engine import RoadmapEngine
from src.cace.confidence_cache import ConfidenceCache


class ApmEngine:
    def __init__(
        self,
        *,
        context_builder: ExecutiveContextBuilder | None = None,
        executive_engine: ExecutiveEngine | None = None,
        policy_engine: PolicyEngine | None = None,
        governance_engine: GovernanceEngine | None = None,
        planner: ExecutivePlanner | None = None,
        opportunity_engine: OpportunityEngine | None = None,
        risk_engine: RiskEngine | None = None,
        roadmap_engine: RoadmapEngine | None = None,
        recommendation_engine: RecommendationEngine | None = None,
        execution_queue: ExecutionQueue | None = None,
        repo: ExecutiveRepository | None = None,
        reporter: ExecutiveReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._context_builder = context_builder or ExecutiveContextBuilder()
        self._executive = executive_engine or ExecutiveEngine()
        self._policy = policy_engine or PolicyEngine()
        self._governance = governance_engine or GovernanceEngine()
        self._planner = planner or ExecutivePlanner()
        self._opportunity = opportunity_engine or OpportunityEngine()
        self._risk = risk_engine or RiskEngine()
        self._roadmap = roadmap_engine or RoadmapEngine()
        self._recommendation = recommendation_engine or RecommendationEngine()
        self._queue = execution_queue or ExecutionQueue()
        self._repo = repo or ExecutiveRepository(owns_connection=self._owns)
        self._reporter = reporter or ExecutiveReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        upstream: dict[str, Any],
        cace_v16_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        mie_report: dict[str, Any],
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_APM_STATUS.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        context = self._context_builder.build(
            profile_id=profile_id,
            upstream=upstream,
            cace_v16_report=cace_v16_report,
            cace_v17_report=cace_v17_report,
            mie_report=mie_report,
        )
        policies = self._policy.evaluate(context)
        governance = self._governance.evaluate(context, policies)
        executive = self._executive.evaluate(context, mie_report)
        opportunities = self._opportunity.evaluate(context)
        risks = self._risk.evaluate(context)
        planned_actions = self._planner.plan(
            context,
            policies=policies,
            governance=governance,
            opportunities=opportunities,
            risks=risks,
        )
        queued_actions = self._queue.enqueue(planned_actions)
        roadmap = self._roadmap.evaluate(queued_actions)
        recommendations = self._recommendation.evaluate(
            context,
            executive=executive,
            actions=queued_actions,
            mie_report=mie_report,
        )

        report = self._reporter.build(
            context=context,
            executive=executive,
            governance=governance,
            policies=policies,
            recommendations=recommendations,
            opportunities=opportunities,
            risks=risks,
            roadmap=roadmap,
            execution_queue=queued_actions,
        )
        report.update(executive)

        if persist:
            self._repo.save_executive_report(profile_id=profile_id, report=report)
            self._repo.save_execution_queue(profile_id=profile_id, actions=queued_actions)
            self._repo.save_roadmap(profile_id=profile_id, roadmap=roadmap)
            self._repo.save_opportunities(profile_id=profile_id, opportunities=opportunities)
            self._repo.save_risk_alerts(profile_id=profile_id, alerts=risks)

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_APM_QUEUE.format(profile_id=profile_id), queued_actions)
            self._cache.set(CACHE_APM_ROADMAP.format(profile_id=profile_id), roadmap)
            self._cache.set(CACHE_APM_OPPORTUNITIES.format(profile_id=profile_id), opportunities)
            self._cache.set(CACHE_APM_RISK_ALERTS.format(profile_id=profile_id), risks)

        self._last_report = report
        return report

    def approve_action(self, *, action_id: str) -> dict[str, Any]:
        action = self._repo.load_action(action_id=action_id)
        if not action:
            return {"error": "action_not_found", "action_id": action_id}
        updated = self._queue.approve_action(action)
        self._repo.update_action_status(action_id=action_id, status=updated["status"])
        return updated

    def reject_action(self, *, action_id: str, reason: str = "") -> dict[str, Any]:
        action = self._repo.load_action(action_id=action_id)
        if not action:
            return {"error": "action_not_found", "action_id": action_id}
        updated = self._queue.reject_action(action, reason=reason)
        self._repo.update_action_status(action_id=action_id, status=updated["status"], reason=reason)
        return updated

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}
