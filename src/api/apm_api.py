"""Public API for Autonomous Portfolio Manager v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.apm_service import ApmService

_svc: ApmService | None = None


def _service() -> ApmService:
    global _svc
    if _svc is None:
        _svc = ApmService(owns_connections=True)
    return _svc


def get_executive_status(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_executive_status(source_path=source_path, profile_id=profile_id)


def get_execution_queue(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_execution_queue(source_path=source_path, profile_id=profile_id)


def get_roadmap(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_roadmap(source_path=source_path, profile_id=profile_id)


def get_opportunities(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_opportunities(source_path=source_path, profile_id=profile_id)


def get_risk_alerts(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_risk_alerts(source_path=source_path, profile_id=profile_id)


def approve_action(*, action_id: str) -> dict[str, Any]:
    return _service().approve_action(action_id=action_id)


def reject_action(*, action_id: str, reason: str = "") -> dict[str, Any]:
    return _service().reject_action(action_id=action_id, reason=reason)


def run_apm_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_apm_cycle(source_path=source_path, profile_id=profile_id)


def close_apm_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
