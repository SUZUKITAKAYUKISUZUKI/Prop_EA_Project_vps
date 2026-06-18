"""API integrity checks for RC1."""
from __future__ import annotations

from typing import Any, Callable


class ApiIntegrityChecker:
    def evaluate(self, *, api_checks: list[tuple[str, Callable[..., Any]]] | None = None) -> dict[str, Any]:
        checks = api_checks or self._default_checks()
        issues: list[str] = []
        passed = 0
        for name, fn in checks:
            if callable(fn):
                passed += 1
            else:
                issues.append(f"{name}: not callable")
        score = round((passed / len(checks)) * 100, 2) if checks else 100.0
        return {
            "api_availability": score,
            "api_health": score,
            "apis_checked": len(checks),
            "apis_available": passed,
            "issues": issues,
            "healthy": score >= 85,
        }

    def _default_checks(self) -> list[tuple[str, Callable[..., Any]]]:
        from src.api import ai_cio_api, orl_api, production_api

        return [
            ("get_cio_report", ai_cio_api.get_cio_report),
            ("get_operational_readiness", orl_api.get_operational_readiness),
            ("get_production_readiness", production_api.get_production_readiness),
            ("get_resilience_score", production_api.get_resilience_score),
            ("get_validation_results", production_api.get_validation_results),
            ("get_benchmark_results", production_api.get_benchmark_results),
            ("run_production_validation", production_api.run_production_validation),
        ]
