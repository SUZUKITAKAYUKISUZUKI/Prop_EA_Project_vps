"""APM v2 service layer."""
from __future__ import annotations

from typing import Any

from src.apm_v2.engine import ApmV2Engine
from src.services.apm_service import ApmService


class ApmV2Service:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._apm = ApmService(owns_connections=owns_connections)
        self._engine = ApmV2Engine(owns_connections=owns_connections)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._apm.close()

    def _run_stack(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        upstream, v16, v17, mie = self._apm._run_intelligence_stack(**kwargs)
        apm_v1 = self._apm._engine.run(
            profile_id=upstream["profile_id"],
            upstream=upstream,
            cace_v16_report=v16,
            cace_v17_report=v17,
            mie_report=mie,
            persist=kwargs.get("persist", False),
            use_cache=not kwargs.get("persist", False),
        )
        return upstream, apm_v1, v17, mie, v16

    def get_executive_board(self, **kwargs: Any) -> dict[str, Any]:
        return self.get_executive_learning(**kwargs)

    def get_executive_learning(self, **kwargs: Any) -> dict[str, Any]:
        upstream, apm_v1, v17, mie, _v16 = self._run_stack(**kwargs)
        return self._engine.run(
            profile_id=upstream["profile_id"],
            apm_v1_report=apm_v1,
            cace_v17_report=v17,
            mie_report=mie,
            upstream=upstream,
            persist=False,
            use_cache=True,
        )

    def get_executive_memory(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.get_executive_learning(**kwargs).get("executive_memory") or [])

    def get_executive_lessons(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.get_executive_learning(**kwargs).get("executive_lessons") or [])

    def get_decision_outcomes(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.get_executive_learning(**kwargs).get("decision_outcomes") or [])

    def get_executive_score(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_executive_learning(**kwargs)
        return {
            "executive_score": report.get("executive_score"),
            "executive_score_v2": report.get("executive_score_v2"),
            "components": report.get("executive_score_components"),
            "decision_accuracy": report.get("decision_accuracy"),
        }

    def get_board_consensus(self, **kwargs: Any) -> dict[str, Any]:
        report = self.get_executive_learning(**kwargs)
        board = report.get("board") or {}
        return {
            "board_consensus": report.get("board_consensus"),
            "agreement_ratio": board.get("agreement_ratio"),
            "average_confidence": board.get("average_confidence"),
            "majority_recommendation": board.get("majority_recommendation"),
            "board_members": board.get("board_members"),
        }

    def run_executive_learning_cycle(self, **kwargs: Any) -> dict[str, Any]:
        upstream = self._apm._mie._load_upstream(**{**kwargs, "persist_age_v4": True})
        v16, v17 = self._apm._mie._run_stack(upstream, persist=True)
        from src.meta_intelligence.engine import MetaIntelligenceEngine

        mie = MetaIntelligenceEngine(owns_connections=False).run(
            profile_id=upstream["profile_id"],
            cace_v16_report=v16,
            cace_v17_report=v17,
            upstream=upstream,
            persist=True,
            use_cache=False,
        )
        apm_v1 = self._apm._engine.run(
            profile_id=upstream["profile_id"],
            upstream=upstream,
            cace_v16_report=v16,
            cace_v17_report=v17,
            mie_report=mie,
            persist=True,
            use_cache=False,
        )
        return self._engine.run(
            profile_id=upstream["profile_id"],
            apm_v1_report=apm_v1,
            cace_v17_report=v17,
            mie_report=mie,
            upstream=upstream,
            persist=True,
            use_cache=False,
        )
