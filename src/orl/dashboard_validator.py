"""Dashboard route validation for ORL v1."""
from __future__ import annotations

from typing import Any


class DashboardValidator:
    REQUIRED_ROUTES = (
        "/dashboard/portfolio_intelligence/api/cio/v1",
        "/dashboard/portfolio_intelligence/api/cio",
        "/dashboard/portfolio_intelligence/api/orl",
        "/dashboard/portfolio_intelligence/api/orl/health",
        "/dashboard/portfolio_intelligence/api/orl/audit",
    )

    def evaluate(self) -> dict[str, Any]:
        issues: list[str] = []
        found = 0
        try:
            from dashboard.portfolio_intelligence_panel import router

            route_paths = {getattr(r, "path", "") for r in router.routes}
            for required in self.REQUIRED_ROUTES:
                suffix = required.replace("/dashboard/portfolio_intelligence", "")
                if suffix in route_paths:
                    found += 1
                else:
                    issues.append(f"Missing route: {suffix}")
        except Exception as exc:
            issues.append(f"Dashboard router load failed: {exc}")
            return {
                "dashboard_health": 0.0,
                "routes_checked": len(self.REQUIRED_ROUTES),
                "routes_present": 0,
                "issues": issues,
                "healthy": False,
            }

        score = round((found / len(self.REQUIRED_ROUTES)) * 100, 2)
        return {
            "dashboard_health": score,
            "routes_checked": len(self.REQUIRED_ROUTES),
            "routes_present": found,
            "issues": issues,
            "healthy": score >= 85,
        }
