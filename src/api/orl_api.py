"""Public API for Operational Readiness Layer v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.orl_service import OrlService

_svc: OrlService | None = None


def _service() -> OrlService:
    global _svc
    if _svc is None:
        _svc = OrlService(owns_connections=True)
    return _svc


def get_operational_readiness(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_operational_readiness(source_path=source_path, profile_id=profile_id)


def get_system_health(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_system_health(source_path=source_path, profile_id=profile_id)


def get_operational_audit(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_operational_audit(source_path=source_path, profile_id=profile_id)


def get_recommendation_consistency(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_recommendation_consistency(source_path=source_path, profile_id=profile_id)


def run_operational_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_operational_cycle(source_path=source_path, profile_id=profile_id)


def close_orl_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
