"""CACE v1.7 service layer."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_intelligence_engine import ConfidenceIntelligenceEngine
from src.cace_v16.confidence_v16_engine import ConfidenceV16Engine
from src.cace_v17.calibration_api import CalibrationEngine
from src.services.cace_service import CaceService


class CaceV17Service:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._cace = CaceService(owns_connections=owns_connections)
        self._v15 = ConfidenceIntelligenceEngine()
        self._v16 = ConfidenceV16Engine(owns_connections=owns_connections)
        self._engine = CalibrationEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._v16.close()
            self._cace.close()

    def _load_upstream(self, **kwargs: Any) -> dict[str, Any]:
        pid, paae, pdts, prae_v2, state, slm, age_v4 = self._cace._load_upstream_and_age_v4(
            source_path=kwargs.get("source_path"),
            profile_id=kwargs.get("profile_id"),
            persist_age_v4=kwargs.get("persist_age_v4", False),
        )
        return {
            "profile_id": pid,
            "paae": paae,
            "pdts": pdts,
            "prae_v2": prae_v2,
            "state_analytics": state,
            "slm": slm,
            "age_v4": age_v4,
        }

    def _run_v1(self, upstream: dict[str, Any], *, persist: bool) -> dict[str, Any]:
        return self._cace._engine.run_confidence_cycle(
            profile_id=upstream["profile_id"],
            paae=upstream["paae"],
            pdts=upstream["pdts"],
            prae_v2=upstream["prae_v2"],
            state_analytics=upstream["state_analytics"],
            slm=upstream["slm"],
            age_v4=upstream["age_v4"],
            persist=persist,
            enrich_age_v4=False,
        )

    def _run_v15(self, upstream: dict[str, Any], v1_report: dict[str, Any], *, persist: bool) -> dict[str, Any]:
        return self._v15.run(
            profile_id=upstream["profile_id"],
            cace_v1_report=v1_report,
            persist=persist,
            use_cache=not persist,
        )

    def _run_v16(self, upstream: dict[str, Any], v15_report: dict[str, Any], *, persist: bool) -> dict[str, Any]:
        return self._v16.run(
            profile_id=upstream["profile_id"],
            cace_v15_report=v15_report,
            upstream=upstream,
            persist=persist,
            use_cache=not persist,
        )

    def get_confidence_calibration(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence_v17(**kwargs)
        return dict(report.get("calibration") or {})

    def get_decision_accuracy(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence_v17(**kwargs)
        return dict(report.get("decision_accuracy") or {})

    def get_confidence_reliability(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence_v17(**kwargs)
        return dict(report.get("reliability") or {})

    def get_recommendation_accuracy(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence_v17(**kwargs)
        return {
            "recommendation_accuracy": report.get("recommendation_accuracy"),
            "detail": report.get("recommendation_accuracy_detail"),
        }

    def get_confidence_learning(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence_v17(**kwargs)
        return dict(report.get("learning") or {})

    def get_confidence_intelligence_v17(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._load_upstream(**kwargs)
        v1 = self._run_v1(upstream, persist=False)
        v15 = self._run_v15(upstream, v1, persist=False)
        v16 = self._run_v16(upstream, v15, persist=False)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            upstream=upstream,
            persist=False,
            use_cache=True,
            capture_new_decision=False,
        )

    def run_calibration_cycle(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._load_upstream(**{**kwargs, "persist_age_v4": True})
        v1 = self._run_v1(upstream, persist=True)
        v15 = self._run_v15(upstream, v1, persist=True)
        v16 = self._run_v16(upstream, v15, persist=True)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            upstream=upstream,
            persist=True,
            use_cache=False,
            capture_new_decision=True,
        )
