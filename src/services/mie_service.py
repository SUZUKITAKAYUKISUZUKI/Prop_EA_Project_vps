"""MIE v1 service layer."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_intelligence_engine import ConfidenceIntelligenceEngine
from src.cace_v16.confidence_v16_engine import ConfidenceV16Engine
from src.cace_v17.calibration_api import CalibrationEngine
from src.meta_intelligence.engine import MetaIntelligenceEngine
from src.services.cace_service import CaceService


class MieService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._cace = CaceService(owns_connections=owns_connections)
        self._v15 = ConfidenceIntelligenceEngine()
        self._v16 = ConfidenceV16Engine(owns_connections=owns_connections)
        self._v17 = CalibrationEngine(owns_connections=owns_connections)
        self._engine = MetaIntelligenceEngine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._v17.close()
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

    def _run_stack(self, upstream: dict[str, Any], *, persist: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        v1 = self._cace._engine.run_confidence_cycle(
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
        v15 = self._v15.run(
            profile_id=upstream["profile_id"],
            cace_v1_report=v1,
            persist=persist,
            use_cache=not persist,
        )
        v16 = self._v16.run(
            profile_id=upstream["profile_id"],
            cace_v15_report=v15,
            upstream=upstream,
            persist=persist,
            use_cache=not persist,
        )
        v17 = self._v17.run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            upstream=upstream,
            persist=persist,
            use_cache=not persist,
            capture_new_decision=persist,
        )
        return v16, v17

    def get_meta_intelligence(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._load_upstream(**kwargs)
        v16, v17 = self._run_stack(upstream, persist=False)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            cace_v17_report=v17,
            upstream=upstream,
            persist=False,
            use_cache=True,
        )

    def get_module_trust_scores(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.get_meta_intelligence(**kwargs).get("module_trust_scores") or {})

    def get_module_rankings(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.get_meta_intelligence(**kwargs).get("module_rankings") or [])

    def get_module_drift(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_meta_intelligence(**kwargs)
        return dict(report.get("drift") or {})

    def get_self_improvement_notes(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.get_meta_intelligence(**kwargs).get("self_improvement_notes") or [])

    def run_meta_intelligence_cycle(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._load_upstream(**{**kwargs, "persist_age_v4": True})
        v16, v17 = self._run_stack(upstream, persist=True)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            cace_v17_report=v17,
            upstream=upstream,
            persist=True,
            use_cache=False,
        )
