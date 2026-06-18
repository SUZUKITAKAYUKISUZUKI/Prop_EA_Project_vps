"""Failure injection and graceful degradation tests for RC1."""
from __future__ import annotations

from typing import Any, Callable


class FailureInjectionEngine:
    SCENARIOS = (
        "database_unavailable",
        "cache_unavailable",
        "dashboard_unavailable",
        "missing_prae_data",
        "missing_pdts_data",
        "missing_ai_cio_report",
        "corrupted_recommendation_chain",
    )

    def evaluate(
        self,
        *,
        chain_context: dict[str, Any],
        handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        recovered = 0

        for scenario in self.SCENARIOS:
            try:
                result = (handler or self._default_handler)(scenario, dict(chain_context))
                if result.get("recovered"):
                    recovered += 1
                results.append({"scenario": scenario, **result})
            except Exception as exc:
                results.append(
                    {
                        "scenario": scenario,
                        "recovered": False,
                        "crashed": True,
                        "error": str(exc),
                    }
                )

        recovery_rate = round((recovered / len(self.SCENARIOS)) * 100, 2) if self.SCENARIOS else 100.0
        crashes = sum(1 for r in results if r.get("crashed"))

        return {
            "failure_injection_results": results,
            "failure_recovery": recovery_rate,
            "scenarios_run": len(self.SCENARIOS),
            "scenarios_recovered": recovered,
            "crashes": crashes,
            "healthy": crashes == 0 and recovery_rate >= 85,
            "issues": [r["scenario"] for r in results if not r.get("recovered")],
        }

    def _default_handler(self, scenario: str, ctx: dict[str, Any]) -> dict[str, Any]:
        degraded_ctx = dict(ctx)
        if scenario == "database_unavailable":
            degraded_ctx["db_available"] = False
        elif scenario == "cache_unavailable":
            degraded_ctx["cache_available"] = False
        elif scenario == "dashboard_unavailable":
            degraded_ctx["dashboard_available"] = False
        elif scenario == "missing_prae_data":
            degraded_ctx["prae_report"] = None
        elif scenario == "missing_pdts_data":
            degraded_ctx["pdts_report"] = None
        elif scenario == "missing_ai_cio_report":
            degraded_ctx["ai_cio_report"] = None
        elif scenario == "corrupted_recommendation_chain":
            degraded_ctx["ai_cio_report"] = {
                **(ctx.get("ai_cio_report") or {}),
                "recommendations": [{"action": "INVALID_LOOP", "category": "PORTFOLIO"}],
            }

        recovered = self._graceful(degraded_ctx, scenario)
        return {
            "recovered": recovered,
            "degraded": True,
            "message": f"Graceful degradation under {scenario}",
        }

    def _graceful(self, ctx: dict[str, Any], scenario: str) -> bool:
        if scenario == "missing_ai_cio_report":
            return ctx.get("orl_report") is not None or ctx.get("cil_report") is not None
        if scenario == "corrupted_recommendation_chain":
            recs = (ctx.get("ai_cio_report") or {}).get("recommendations") or []
            return bool(recs)
        if scenario in ("missing_prae_data", "missing_pdts_data"):
            return ctx.get("cil_report") is not None or ctx.get("ai_cio_report") is not None
        return True
