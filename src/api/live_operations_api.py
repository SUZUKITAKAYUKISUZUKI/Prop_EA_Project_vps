"""Public API for RC2 Live Operations Layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.live_operations_service import LiveOperationsService

_svc: LiveOperationsService | None = None


def _service() -> LiveOperationsService:
    global _svc
    if _svc is None:
        _svc = LiveOperationsService(owns_connections=True)
    return _svc


def get_daily_briefing(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_daily_briefing(source_path=source_path, profile_id=profile_id)


def get_daily_digest(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_daily_digest(source_path=source_path, profile_id=profile_id)


def get_operational_alerts(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_operational_alerts(source_path=source_path, profile_id=profile_id)


def get_operational_score(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_operational_score(source_path=source_path, profile_id=profile_id)


def get_live_readiness(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_live_readiness(source_path=source_path, profile_id=profile_id)


def run_live_operations_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_live_operations_cycle(source_path=source_path, profile_id=profile_id)


def close_live_operations_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
