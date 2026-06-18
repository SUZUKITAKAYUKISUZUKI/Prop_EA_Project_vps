"""API for CACE v1.5 Confidence Intelligence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.cace_v15_service import CaceV15Service

_svc: CaceV15Service | None = None


def _service() -> CaceV15Service:
    global _svc
    if _svc is None:
        _svc = CaceV15Service(owns_connections=True)
    return _svc


def get_confidence_intelligence(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_intelligence(source_path=source_path, profile_id=profile_id)


def get_confidence_breakdown(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, float]:
    return _service().get_confidence_breakdown(source_path=source_path, profile_id=profile_id)


def get_confidence_trend(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_trend(source_path=source_path, profile_id=profile_id)


def get_regime_confidence(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_regime_confidence(source_path=source_path, profile_id=profile_id)


def get_confidence_history(
    *,
    profile_id: str | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    return _service().get_confidence_history(profile_id=profile_id, limit=limit)


def run_confidence_intelligence_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_confidence_intelligence_cycle(source_path=source_path, profile_id=profile_id)


def close_cace_v15_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
