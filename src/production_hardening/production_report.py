"""Production readiness report builder for RC1."""
from __future__ import annotations

from typing import Any


class ProductionReport:
    def build(
        self,
        *,
        profile_id: str,
        validation: dict[str, Any],
        resilience: dict[str, Any],
        end_to_end: dict[str, Any],
        chain: dict[str, Any],
        database: dict[str, Any],
        api: dict[str, Any],
        dashboard: dict[str, Any],
        cache: dict[str, Any],
        cio: dict[str, Any],
        portfolio: dict[str, Any],
        failures: dict[str, Any],
        benchmarks: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            **validation,
            **resilience,
            "end_to_end": end_to_end,
            "recommendation_chain": chain,
            "database": database,
            "api": api,
            "dashboard": dashboard,
            "cache": cache,
            "cio_consistency": cio,
            "portfolio_consistency": portfolio,
            "failure_injection": failures,
            "performance": benchmarks,
            "operational_metrics": metrics,
            "validation_results": {
                "end_to_end_validation": end_to_end.get("end_to_end_validation"),
                "recommendation_chain_health": chain.get("recommendation_chain_health"),
                "data_integrity": database.get("data_integrity"),
                "api_health": api.get("api_health"),
                "dashboard_health": dashboard.get("dashboard_health"),
                "cache_integrity": cache.get("cache_integrity"),
            },
            "benchmark_results": benchmarks.get("benchmarks"),
            "summary": self._summary(validation, resilience),
        }

    def _summary(self, validation: dict[str, Any], resilience: dict[str, Any]) -> str:
        pr = validation.get("production_readiness_score", 0)
        rs = resilience.get("resilience_score", 0)
        status = validation.get("production_status", "NOT_READY")
        if validation.get("production_ready"):
            return f"Portfolio OS RC1 {status}: production readiness {pr}, resilience {rs}."
        return f"Portfolio OS RC1 {status}: readiness {pr}, resilience {rs}. Review open production issues."
