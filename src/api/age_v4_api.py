"""API for AGE v4 Strategic Decision Simulator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.strategic_governor_service import StrategicGovernorService

_svc: StrategicGovernorService | None = None


def _service() -> StrategicGovernorService:
    global _svc
    if _svc is None:
        _svc = StrategicGovernorService(owns_connections=True)
    return _svc


def get_strategic_report(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_strategic_report(
        source_path=source_path,
        profile_id=profile_id,
        persist=False,
    )


def get_future_scenarios(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_future_scenarios(source_path=source_path, profile_id=profile_id)


def get_future_rankings(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_future_rankings(source_path=source_path, profile_id=profile_id)


def get_best_future(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_best_future(source_path=source_path, profile_id=profile_id)


def run_strategic_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_strategic_cycle(source_path=source_path, profile_id=profile_id)


def close_age_v4_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
