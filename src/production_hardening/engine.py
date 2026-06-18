"""Portfolio OS RC1 production hardening orchestrator."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.production_hardening.api_integrity_checker import ApiIntegrityChecker
from src.production_hardening.cache_integrity_checker import CacheIntegrityChecker
from src.production_hardening.cio_consistency_checker import CioConsistencyChecker
from src.production_hardening.config import CACHE_PRODUCTION_BENCHMARKS, CACHE_PRODUCTION_READINESS
from src.production_hardening.dashboard_integrity_checker import DashboardIntegrityChecker
from src.production_hardening.database_integrity_checker import DatabaseIntegrityChecker
from src.production_hardening.end_to_end_validator import EndToEndValidator
from src.production_hardening.failure_injection_engine import FailureInjectionEngine
from src.production_hardening.operational_metrics import OperationalMetrics
from src.production_hardening.performance_benchmark import PerformanceBenchmark
from src.production_hardening.portfolio_consistency_checker import PortfolioConsistencyChecker
from src.production_hardening.production_report import ProductionReport
from src.production_hardening.production_repository import ProductionRepository
from src.production_hardening.production_validator import ProductionValidator
from src.production_hardening.recommendation_chain_validator import RecommendationChainValidator
from src.production_hardening.resilience_tester import ResilienceTester


class ProductionHardeningEngine:
    """RC1 production validation — no new intelligence, operational confidence only."""

    def __init__(
        self,
        *,
        database_checker: DatabaseIntegrityChecker | None = None,
        api_checker: ApiIntegrityChecker | None = None,
        cache_checker: CacheIntegrityChecker | None = None,
        dashboard_checker: DashboardIntegrityChecker | None = None,
        end_to_end: EndToEndValidator | None = None,
        chain_validator: RecommendationChainValidator | None = None,
        portfolio_checker: PortfolioConsistencyChecker | None = None,
        cio_checker: CioConsistencyChecker | None = None,
        failure_engine: FailureInjectionEngine | None = None,
        resilience_tester: ResilienceTester | None = None,
        benchmark: PerformanceBenchmark | None = None,
        production_validator: ProductionValidator | None = None,
        metrics_engine: OperationalMetrics | None = None,
        repo: ProductionRepository | None = None,
        reporter: ProductionReport | None = None,
        cache: ConfidenceCache | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._database = database_checker or DatabaseIntegrityChecker()
        self._api = api_checker or ApiIntegrityChecker()
        self._cache_checker = cache_checker or CacheIntegrityChecker()
        self._dashboard = dashboard_checker or DashboardIntegrityChecker()
        self._end_to_end = end_to_end or EndToEndValidator()
        self._chain = chain_validator or RecommendationChainValidator()
        self._portfolio = portfolio_checker or PortfolioConsistencyChecker()
        self._cio = cio_checker or CioConsistencyChecker()
        self._failures = failure_engine or FailureInjectionEngine()
        self._resilience = resilience_tester or ResilienceTester()
        self._benchmark = benchmark or PerformanceBenchmark()
        self._validator = production_validator or ProductionValidator()
        self._metrics = metrics_engine or OperationalMetrics()
        self._repo = repo or ProductionRepository(owns_connection=self._owns)
        self._reporter = reporter or ProductionReport()
        self._cache = cache or ConfidenceCache()
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def run(
        self,
        *,
        profile_id: str,
        chain_context: dict[str, Any],
        persist: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = CACHE_PRODUCTION_READINESS.format(profile_id=profile_id)
        if use_cache and not persist:
            cached = self._cache.get_if_fresh(cache_key)
            if cached:
                self._last_report = cached
                return cached

        chain_context = {**chain_context, "profile_id": profile_id}

        database = self._database.evaluate(profile_id=profile_id)
        api = self._api.evaluate()
        cache_result = self._cache_checker.evaluate(profile_id=profile_id)
        dashboard = self._dashboard.evaluate()
        end_to_end = self._end_to_end.evaluate(chain_context=chain_context)
        chain = self._chain.evaluate(chain_context=chain_context)
        portfolio = self._portfolio.evaluate(
            profile_id=profile_id,
            ai_cio_report=chain_context.get("ai_cio_report"),
            cil_report=chain_context.get("cil_report"),
            chain_context=chain_context,
        )
        cio = self._cio.evaluate(
            ai_cio_report=chain_context.get("ai_cio_report"),
            orl_report=chain_context.get("orl_report"),
        )
        failures = self._failures.evaluate(chain_context=chain_context)
        benchmarks = self._benchmark.evaluate(profile_id=profile_id)

        resilience_input = {
            "data_integrity": database.get("data_integrity"),
            "api_availability": api.get("api_availability"),
            "dashboard_availability": dashboard.get("dashboard_availability"),
            "recommendation_consistency": chain.get("recommendation_consistency"),
            "ai_cio_availability": cio.get("ai_cio_availability"),
            "failure_recovery": failures.get("failure_recovery"),
        }
        resilience = self._resilience.evaluate(resilience_input)

        validation = self._validator.evaluate(
            resilience=resilience,
            end_to_end=end_to_end,
            chain=chain,
            database=database,
            api=api,
            dashboard=dashboard,
            cache=cache_result,
            cio=cio,
            portfolio=portfolio,
            failures=failures,
            benchmarks=benchmarks,
        )

        metrics = self._metrics.evaluate(
            validation=validation,
            resilience=resilience,
            benchmarks=benchmarks,
            end_to_end=end_to_end,
            chain=chain,
            failures=failures,
        )

        report = self._reporter.build(
            profile_id=profile_id,
            validation=validation,
            resilience=resilience,
            end_to_end=end_to_end,
            chain=chain,
            database=database,
            api=api,
            dashboard=dashboard,
            cache=cache_result,
            cio=cio,
            portfolio=portfolio,
            failures=failures,
            benchmarks=benchmarks,
            metrics=metrics,
        )

        if persist:
            self._repo.save_readiness(profile_id=profile_id, report=report)
            self._repo.save_validation_history(profile_id=profile_id, report=report)
            self._repo.save_benchmark_history(profile_id=profile_id, benchmarks=benchmarks)
            self._repo.save_resilience_history(
                profile_id=profile_id,
                resilience_score=float(resilience.get("resilience_score") or 0),
                failure_recovery=float(failures.get("failure_recovery") or 0),
            )
            for item in failures.get("failure_injection_results") or []:
                self._repo.save_failure(
                    profile_id=profile_id,
                    scenario=str(item.get("scenario") or ""),
                    recovered=bool(item.get("recovered")),
                    payload=item,
                )

        if use_cache or persist:
            self._cache.set(cache_key, report)
            self._cache.set(CACHE_PRODUCTION_BENCHMARKS.format(profile_id=profile_id), benchmarks)

        self._last_report = report
        return report

    def get_last_report(self) -> dict[str, Any]:
        return self._last_report or {}
