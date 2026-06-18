"""CACE top-level engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.cace.confidence_engine import ConfidenceEngine
from src.cace.confidence_repository import ConfidenceRepository


class CaceEngine:
    """Capital Allocation Confidence Engine — evaluation only, never places trades."""

    def __init__(
        self,
        *,
        confidence: ConfidenceEngine | None = None,
        repo: ConfidenceRepository | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections
        self._confidence = confidence or ConfidenceEngine(owns_connections=owns_connections)
        self._repo = repo or ConfidenceRepository(owns_connection=False)
        self._last_report: dict[str, Any] | None = None

    def close(self) -> None:
        self._confidence.close()
        if self._owns:
            self._repo.close()

    def run_confidence_cycle(
        self,
        *,
        profile_id: str,
        paae: dict[str, Any],
        pdts: dict[str, Any],
        prae_v2: dict[str, Any],
        state_analytics: dict[str, Any],
        slm: dict[str, Any],
        age_v4: dict[str, Any],
        current_weights: dict[str, float] | None = None,
        persist: bool = True,
        enrich_age_v4: bool = True,
    ) -> dict[str, Any]:
        report = self._confidence.run(
            profile_id=profile_id,
            paae=paae,
            pdts=pdts,
            prae_v2=prae_v2,
            state_analytics=state_analytics,
            slm=slm,
            age_v4=age_v4,
            current_weights=current_weights,
            persist=persist,
            use_cache=not persist,
        )
        if enrich_age_v4:
            report["age_v4_enriched"] = self._confidence.enrich_age_v4_report(age_v4, report)
        self._last_report = report
        return report

    def get_allocation_confidence(self) -> dict[str, Any]:
        report = self.get_confidence_report()
        return report.get("allocation_confidence") or {}

    def get_strategy_confidence(self) -> list[dict[str, Any]]:
        report = self.get_confidence_report()
        return report.get("strategy_confidence") or []

    def get_portfolio_confidence(self) -> dict[str, Any]:
        report = self.get_confidence_report()
        return report.get("portfolio_confidence") or {}

    def get_confidence_report(self) -> dict[str, Any]:
        return self._last_report or self._confidence.get_last_report() or {}

    def get_confidence_history(self, *, profile_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        return self._repo.list_confidence_history(profile_id=profile_id, limit=limit)
