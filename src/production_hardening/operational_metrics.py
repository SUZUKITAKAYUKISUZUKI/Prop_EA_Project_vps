"""Operational metrics aggregation for RC1."""
from __future__ import annotations

from typing import Any


class OperationalMetrics:
    def evaluate(
        self,
        *,
        validation: dict[str, Any],
        resilience: dict[str, Any],
        benchmarks: dict[str, Any],
        end_to_end: dict[str, Any],
        chain: dict[str, Any],
        failures: dict[str, Any],
    ) -> dict[str, Any]:
        critical = [
            i
            for i in (validation.get("open_issues") or [])
            if any(k in i.lower() for k in ("missing", "conflict", "crash", "failed", "corrupt"))
        ]
        return {
            "layers_validated": sum(1 for v in (end_to_end.get("layers_present") or {}).values() if v),
            "layers_total": len(end_to_end.get("layers_present") or {}),
            "recommendation_chain_health": chain.get("recommendation_chain_health"),
            "failure_recovery_rate": failures.get("failure_recovery"),
            "benchmark_score": benchmarks.get("benchmark_score"),
            "critical_issue_count": len(critical),
            "open_production_issues": validation.get("open_issues") or [],
        }
