"""API for Capital Allocation Confidence Engine (CACE)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.cace_service import CaceService

_svc: CaceService | None = None


def _service() -> CaceService:
    global _svc
    if _svc is None:
        _svc = CaceService(owns_connections=True)
    return _svc


def get_allocation_confidence(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_allocation_confidence(source_path=source_path, profile_id=profile_id)


def get_strategy_confidence(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_strategy_confidence(source_path=source_path, profile_id=profile_id)


def get_portfolio_confidence(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_portfolio_confidence(source_path=source_path, profile_id=profile_id)


def get_confidence_history(
    *,
    profile_id: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    return _service().get_confidence_history(profile_id=profile_id, limit=limit)


def run_confidence_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_confidence_cycle(source_path=source_path, profile_id=profile_id)


def close_cace_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
