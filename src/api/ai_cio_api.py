"""Public API for AI CIO v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.ai_cio_service import AiCioService

_svc: AiCioService | None = None


def _service() -> AiCioService:
    global _svc
    if _svc is None:
        _svc = AiCioService(owns_connections=True)
    return _svc


def get_cio_report(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_cio_report(source_path=source_path, profile_id=profile_id)


def get_cio_opinion(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_cio_opinion(source_path=source_path, profile_id=profile_id)


def get_cio_recommendations(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_cio_recommendations(source_path=source_path, profile_id=profile_id)


def get_cio_history(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_cio_history(source_path=source_path, profile_id=profile_id)


def get_cio_memory(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_cio_memory(source_path=source_path, profile_id=profile_id)


def run_cio_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_cio_cycle(source_path=source_path, profile_id=profile_id)


def record_cio_actual_outcome(
    *,
    actual_outcome: dict[str, Any],
    source_path: str | Path | None = None,
    profile_id: str | None = None,
    report_id: int | None = None,
) -> dict[str, Any]:
    return _service().record_actual_outcome(
        actual_outcome=actual_outcome,
        source_path=source_path,
        profile_id=profile_id,
        report_id=report_id,
    )


def close_ai_cio_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
