"""API availability validation for ORL v1."""
from __future__ import annotations

from typing import Any, Callable


class ApiValidator:
    def evaluate(self, *, api_checks: list[tuple[str, Callable[..., Any]]] | None = None) -> dict[str, Any]:
        checks = api_checks or self._default_checks()
        issues: list[str] = []
        passed = 0

        for name, fn in checks:
            try:
                if not callable(fn):
                    issues.append(f"{name}: not callable")
                    continue
                passed += 1
            except Exception as exc:
                issues.append(f"{name}: {exc}")

        score = round((passed / len(checks)) * 100, 2) if checks else 100.0
        return {
            "api_health": score,
            "apis_checked": len(checks),
            "apis_available": passed,
            "issues": issues,
            "healthy": score >= 85,
        }

    def _default_checks(self) -> list[tuple[str, Callable[..., Any]]]:
        from src.api import ai_cio_api, cio_intelligence_api, orl_api

        return [
            ("get_cio_report", ai_cio_api.get_cio_report),
            ("get_cio_intelligence", cio_intelligence_api.get_cio_intelligence),
            ("get_operational_readiness", orl_api.get_operational_readiness),
            ("get_system_health", orl_api.get_system_health),
            ("get_operational_audit", orl_api.get_operational_audit),
            ("get_recommendation_consistency", orl_api.get_recommendation_consistency),
            ("run_operational_cycle", orl_api.run_operational_cycle),
        ]
