"""CACE v1.6 service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.cace.confidence_intelligence_engine import ConfidenceIntelligenceEngine
from src.cace_v16.confidence_v16_engine import ConfidenceV16Engine
from src.services.cace_service import CaceService


class CaceV16Service:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._cace = CaceService(owns_connections=owns_connections)
        self._v15 = ConfidenceIntelligenceEngine()
        self._engine = ConfidenceV16Engine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
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

    def get_confidence_intelligence_v16(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._load_upstream(**kwargs)
        v1 = self._run_v1(upstream, persist=False)
        v15 = self._run_v15(upstream, v1, persist=False)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            cace_v15_report=v15,
            upstream=upstream,
            persist=False,
            use_cache=True,
        )

    def get_confidence_decay(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence_v16(**kwargs)
        return dict(report.get("decay") or {})

    def get_confidence_consensus(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_confidence_intelligence_v16(**kwargs)
        return dict(report.get("consensus") or {})

    def run_confidence_v16_cycle(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._load_upstream(**{**kwargs, "persist_age_v4": True})
        v1 = self._run_v1(upstream, persist=True)
        v15 = self._run_v15(upstream, v1, persist=True)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            cace_v15_report=v15,
            upstream=upstream,
            persist=True,
            use_cache=False,
        )
