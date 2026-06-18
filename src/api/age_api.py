"""Dashboard API for AI Governor Engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.ai_governor_service import AiGovernorService

_svc: AiGovernorService | None = None


def _service() -> AiGovernorService:
    global _svc
    if _svc is None:
        _svc = AiGovernorService(owns_connections=True)
    return _svc


def get_governor_status(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_governor_status(source_path=source_path, profile_id=profile_id)


def get_governor_report(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_governor_report(source_path=source_path, profile_id=profile_id, persist=False)


def get_governor_recommendations(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
    status: str = "OPEN",
) -> list[dict[str, Any]]:
    return _service().get_governor_recommendations(
        source_path=source_path,
        profile_id=profile_id,
        status=status,
    )


def run_governor_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_governor_cycle(source_path=source_path, profile_id=profile_id)


def get_governor_decision_history(
    *,
    profile_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _service().get_governor_decision_history(profile_id=profile_id, limit=limit)


def close_age_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
