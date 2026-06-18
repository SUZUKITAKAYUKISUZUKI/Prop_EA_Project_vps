"""CACE v1.5 service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.cace.confidence_intelligence_engine import ConfidenceIntelligenceEngine
from src.services.cace_service import CaceService


class CaceV15Service:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._cace_v1 = CaceService(owns_connections=owns_connections)
        self._intelligence = ConfidenceIntelligenceEngine()

    def close(self) -> None:
        if self._owns:
            self._cace_v1.close()

    def _run_v1(self, **kwargs: Any) -> dict[str, Any]:
        return self._cace_v1.get_confidence_report(**kwargs)

    def get_confidence_intelligence(self, **kwargs: Any) -> dict[str, Any]:
        v1 = self._run_v1(**kwargs)
        return self._intelligence.run(
            profile_id=str(v1.get("profile_id") or kwargs.get("profile_id") or "unknown"),
            cace_v1_report=v1,
            persist=False,
            use_cache=True,
        )

    def get_confidence_breakdown(self, **kwargs: Any) -> dict[str, float]:
        report = self.get_confidence_intelligence(**kwargs)
        return dict(report.get("breakdown") or {})

    def get_confidence_trend(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence(**kwargs)
        return {
            "confidence": report.get("confidence"),
            "trend": report.get("trend"),
            "trend_strength": report.get("trend_strength"),
            "trend_direction": report.get("trend_direction"),
            "trend_category": report.get("trend_category"),
            "trend_windows": report.get("trend_windows"),
            "confidence_evolution": report.get("confidence_evolution"),
        }

    def get_regime_confidence(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence(**kwargs)
        return {
            "regime": report.get("regime"),
            "confidence_modifier": report.get("regime_modifier"),
            "raw_confidence": report.get("raw_confidence"),
            "adjusted_confidence": report.get("confidence"),
            "regime_metrics": report.get("regime_metrics"),
            "regime_rationale": report.get("regime_rationale"),
            "regime_appropriate": report.get("regime_appropriate"),
        }

    def get_confidence_history(self, *, profile_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        return self._cace_v1.get_confidence_history(profile_id=profile_id, limit=limit)

    def run_confidence_intelligence_cycle(self, **kwargs: Any) -> dict[str, Any]:
        v1 = self._cace_v1.run_confidence_cycle(**kwargs)
        return self._intelligence.run(
            profile_id=str(v1.get("profile_id") or "unknown"),
            cace_v1_report=v1,
            persist=True,
            use_cache=False,
        )
