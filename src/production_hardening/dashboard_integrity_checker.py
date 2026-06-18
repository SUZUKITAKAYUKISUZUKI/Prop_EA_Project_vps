"""Dashboard integrity checks for RC1."""
from __future__ import annotations

from typing import Any


class DashboardIntegrityChecker:
    REQUIRED_ROUTES = (
        "/api/cio/v1",
        "/api/orl",
        "/api/production",
        "/api/production/health",
        "/api/production/benchmarks",
    )

    def evaluate(self) -> dict[str, Any]:
        issues: list[str] = []
        found = 0
        try:
            from dashboard.portfolio_intelligence_panel import router

            paths = {getattr(r, "path", "") for r in router.routes}
            for route in self.REQUIRED_ROUTES:
                if route in paths:
                    found += 1
                else:
                    issues.append(f"Missing dashboard route: {route}")
        except Exception as exc:
            issues.append(f"Dashboard load failed: {exc}")
            return {
                "dashboard_availability": 0.0,
                "dashboard_health": 0.0,
                "routes_present": 0,
                "issues": issues,
                "healthy": False,
            }

        score = round((found / len(self.REQUIRED_ROUTES)) * 100, 2)
        return {
            "dashboard_availability": score,
            "dashboard_health": score,
            "routes_checked": len(self.REQUIRED_ROUTES),
            "routes_present": found,
            "issues": issues,
            "healthy": score >= 85,
        }
