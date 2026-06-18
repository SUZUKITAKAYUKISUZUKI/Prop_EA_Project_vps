"""ORL v1 orchestration — production readiness validation."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.orl.api_validator import ApiValidator
from src.orl.cache_validator import CacheValidator
from src.orl.config import CACHE_ORL_HEALTH, CACHE_ORL_READINESS
from src.orl.dashboard_validator import DashboardValidator
from src.orl.database_validator import DatabaseValidator
from src.orl.dependency_validator import DependencyValidator
from src.orl.executive_summary_engine import ExecutiveSummaryEngine
from src.orl.operational_report import OperationalReport
from src.orl.operational_repository import OperationalRepository
from src.orl.portfolio_audit_engine import PortfolioAuditEngine
from src.orl.readiness_engine import ReadinessEngine
from src.orl.recommendation_validator import RecommendationValidator
from src.orl.system_health_engine import SystemHealthEngine


class OrlEngine:
    """Validates Portfolio OS operational readiness for daily management."""

    def __init__(
        self,
        *,
        database_validator: DatabaseValidator | None = None,
        api_validator: ApiValidator | None = None,
        cache_validator: CacheValidator | None = None,
        dashboard_validator: DashboardValidator | None = None,
        dependency_validator: DependencyValidator | None = None,
        recommendation_validator: RecommendationValidator | None = None,
        portfolio_audit: PortfolioAuditEngine | None = None,
        system_health_engine: SystemHealthEngine | None = None,
        readiness_engine: ReadinessEngine | None = None,
        executive_engine: ExecutiveSummaryEngine | None = None,
        repo: OperationalRepository | None = None,
        reporter: OperationalReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._database = database_validator or DatabaseValidator()
        self._api = api_validator or ApiValidator()
        self._cache_validator = cache_validator or CacheValidator()
        self._dashboard = dashboard_validator or DashboardValidator()
        self._dependency = dependency_validator or DependencyValidator()
        self._consistency = recommendation_validator or RecommendationValidator()
        self._audit = portfolio_audit or PortfolioAuditEngine()
        self._system_health = system_health_engine or SystemHealthEngine()
        self._readiness = readiness_engine or ReadinessEngine()
        self._executive = executive_engine or ExecutiveSummaryEngine()
        self._repo = repo or OperationalRepository(owns_connection=self._owns)
        self._reporter = reporter or OperationalReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        ai_cio_report: dict[str, Any] | None = None,
        cil_report: dict[str, Any] | None = None,
        apm_report: dict[str, Any] | None = None,
        age_report: dict[str, Any] | None = None,
        cace_report: dict[str, Any] | None = None,
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_ORL_READINESS.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        database = self._database.evaluate(profile_id=profile_id)
        api = self._api.evaluate()
        cache = self._cache_validator.evaluate(profile_id=profile_id)
        dashboard = self._dashboard.evaluate()
        dependency = self._dependency.evaluate()

        ai_cio_check = self._validate_ai_cio(ai_cio_report)
        consistency = self._consistency.evaluate(
            ai_cio_report=ai_cio_report,
            apm_report=apm_report,
            age_report=age_report,
            cace_report=cace_report,
        )
        audit = self._audit.evaluate(
            profile_id=profile_id,
            ai_cio_report=ai_cio_report,
            cil_report=cil_report,
        )

        historical_stability = self._repo.historical_stability_score(profile_id=profile_id)

        component_scores = {
            "database_health": database.get("database_health"),
            "api_health": api.get("api_health"),
            "dashboard_health": dashboard.get("dashboard_health"),
            "cache_health": cache.get("cache_health"),
            "dependency_health": dependency.get("dependency_health"),
            "ai_cio_availability": ai_cio_check.get("ai_cio_availability"),
            "recommendation_consistency": consistency.get("recommendation_consistency"),
            "historical_stability": historical_stability,
        }
        system_health = self._system_health.evaluate(component_scores)

        all_issues: list[str] = []
        for block in (database, api, cache, dashboard, dependency, ai_cio_check, consistency, audit):
            all_issues.extend(block.get("issues") or [])

        critical_issues = [i for i in all_issues if "missing" in i.lower() or "conflict" in i.lower() or "failed" in i.lower()]
        readiness_input = {
            **component_scores,
            "system_health": system_health.get("system_health"),
            "critical_issues": critical_issues,
        }
        readiness = self._readiness.evaluate(readiness_input)
        executive = self._executive.evaluate(
            readiness=readiness,
            system_health=system_health,
            consistency=consistency,
            audit=audit,
            all_issues=all_issues,
        )

        report = self._reporter.build(
            profile_id=profile_id,
            readiness=readiness,
            system_health=system_health,
            database=database,
            api=api,
            dashboard=dashboard,
            cache=cache,
            consistency=consistency,
            audit=audit,
            ai_cio=ai_cio_check,
            executive=executive,
            historical_stability=historical_stability,
        )

        if persist:
            self._repo.save_readiness(profile_id=profile_id, report=report)
            self._repo.save_health_snapshot(
                profile_id=profile_id,
                system_health=float(system_health.get("system_health") or 0),
                readiness_score=float(readiness.get("readiness_score") or 0),
            )
            for issue in all_issues[:10]:
                severity = "critical" if issue in critical_issues else "warning"
                self._repo.save_audit_log(
                    profile_id=profile_id,
                    category="operational_cycle",
                    message=issue,
                    severity=severity,
                )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_ORL_HEALTH.format(profile_id=profile_id), extract_health(report))

        self._last_report = report
        return report

    def _validate_ai_cio(self, report: dict[str, Any] | None) -> dict[str, Any]:
        issues: list[str] = []
        if not report:
            return {"ai_cio_availability": 0.0, "issues": ["AI CIO report missing"], "healthy": False}

        required = ("cio_score", "cio_opinion", "recommendations", "top_opportunity", "top_risk")
        present = sum(1 for key in required if report.get(key) is not None)
        if not report.get("recommendations"):
            issues.append("AI CIO recommendations not visible")
        if report.get("cio_score") is None:
            issues.append("AI CIO score missing")

        score = round((present / len(required)) * 100, 2)
        return {
            "ai_cio_availability": score,
            "issues": issues,
            "healthy": score >= 85 and not issues,
            "report_keys_present": present,
        }

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}


from src.orl.operational_api import extract_health
