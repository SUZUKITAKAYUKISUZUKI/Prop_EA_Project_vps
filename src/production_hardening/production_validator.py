"""Production validation orchestration for RC1."""
from __future__ import annotations

from typing import Any

from src.production_hardening.config import READINESS_THRESHOLDS


class ProductionValidator:
    def evaluate(
        self,
        *,
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
    ) -> dict[str, Any]:
        open_issues: list[str] = []
        for block in (database, api, dashboard, cache, cio, portfolio, end_to_end, chain, failures, benchmarks):
            open_issues.extend(block.get("issues") or [])

        critical = [
            i
            for i in open_issues
            if any(k in i.lower() for k in ("missing", "conflict", "crash", "failed", "corrupt", "integrity"))
        ]

        resilience_score = float(resilience.get("resilience_score") or 0)
        e2e_score = float(end_to_end.get("end_to_end_score") or 0)
        production_readiness = round((resilience_score * 0.6 + e2e_score * 0.4), 2)
        status = self._status(production_readiness)

        production_ready = (
            production_readiness >= 85
            and resilience_score >= 85
            and not critical
            and end_to_end.get("end_to_end_validation")
            and cio.get("healthy")
        )

        return {
            "production_readiness": production_readiness,
            "production_readiness_score": production_readiness,
            "production_status": status,
            "production_ready": production_ready,
            "open_issues": list(dict.fromkeys(open_issues))[:30],
            "critical_issues": critical,
            "open_production_issues": list(dict.fromkeys(open_issues))[:20],
        }

    def _status(self, score: float) -> str:
        for threshold, label in READINESS_THRESHOLDS:
            if score >= threshold:
                return label
        return "NOT_READY"
