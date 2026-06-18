"""Public API for APM v2 Executive Board & Portfolio Memory."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.apm_v2_service import ApmV2Service

_svc: ApmV2Service | None = None


def _service() -> ApmV2Service:
    global _svc
    if _svc is None:
        _svc = ApmV2Service(owns_connections=True)
    return _svc


def get_executive_memory(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_executive_memory(source_path=source_path, profile_id=profile_id)


def get_executive_lessons(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_executive_lessons(source_path=source_path, profile_id=profile_id)


def get_decision_outcomes(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_decision_outcomes(source_path=source_path, profile_id=profile_id)


def get_executive_score(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_executive_score(source_path=source_path, profile_id=profile_id)


def get_board_consensus(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_board_consensus(source_path=source_path, profile_id=profile_id)


def get_executive_board(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_executive_learning(source_path=source_path, profile_id=profile_id)


def run_executive_learning_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_executive_learning_cycle(source_path=source_path, profile_id=profile_id)


def close_apm_v2_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
