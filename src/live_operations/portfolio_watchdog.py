"""Portfolio watchdog for RC2 live operations."""
from __future__ import annotations

from typing import Any


class PortfolioWatchdog:
    CHECKS = (
        "ai_cio_stopped",
        "api_stopped",
        "db_corruption",
        "cache_corruption",
        "recommendation_conflict",
        "abnormal_dd_increase",
        "strategy_degradation",
        "recommendation_loop",
    )

    def evaluate(self, *, ops_context: dict[str, Any]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        ai_cio = ops_context.get("ai_cio_report") or {}
        production = ops_context.get("production_report") or {}
        orl = ops_context.get("orl_report") or {}

        if not ai_cio.get("cio_opinion"):
            issues.append(self._issue("ai_cio_stopped", "CRITICAL", "AI CIO report unavailable"))
        if float(production.get("api_health") or production.get("validation_results", {}).get("api_health") or 100) < 70:
            issues.append(self._issue("api_stopped", "CRITICAL", "API health below threshold"))
        db_health = float(
            production.get("database_health")
            or (production.get("validation_results") or {}).get("data_integrity")
            or 100
        )
        if db_health < 70:
            issues.append(self._issue("db_corruption", "CRITICAL", "Database integrity degraded"))
        cache_health = float(production.get("cache_health") or 100)
        if cache_health < 70:
            issues.append(self._issue("cache_corruption", "WARNING", "Cache integrity degraded"))

        consistency = production.get("recommendation_chain") or orl.get("consistency") or {}
        if consistency.get("contradictions"):
            issues.append(
                self._issue(
                    "recommendation_conflict",
                    "WARNING",
                    f"Recommendation conflicts: {consistency.get('contradictions')}",
                )
            )
        if consistency.get("loops"):
            issues.append(
                self._issue("recommendation_loop", "WARNING", f"Recommendation loops: {consistency.get('loops')}")
            )

        health = float((ai_cio.get("confidence") or 0))
        cil = ops_context.get("cil_report") or {}
        portfolio_health = float(
            cil.get("portfolio_health")
            or (cil.get("executive_components") or {}).get("portfolio_health")
            or 80
        )
        if portfolio_health < 55:
            issues.append(self._issue("abnormal_dd_increase", "WARNING", f"Portfolio health low: {portfolio_health}"))

        mie_trust = float(cil.get("intelligence_trust") or 80)
        if mie_trust < 50:
            issues.append(self._issue("strategy_degradation", "NOTICE", f"Intelligence trust declining: {mie_trust}"))

        score = max(0.0, 100.0 - len(issues) * 12)
        return {
            "watchdog_score": round(score, 2),
            "issues": issues,
            "healthy": not any(i["severity"] == "CRITICAL" for i in issues),
            "checks_run": len(self.CHECKS),
        }

    def _issue(self, check: str, severity: str, message: str) -> dict[str, Any]:
        return {"check": check, "severity": severity, "message": message, "category": severity}
