"""API for CACE v1.6 Confidence Intelligence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.cace_v16_service import CaceV16Service

_svc: CaceV16Service | None = None


def _service() -> CaceV16Service:
    global _svc
    if _svc is None:
        _svc = CaceV16Service(owns_connections=True)
    return _svc


def get_confidence_decay(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_decay(source_path=source_path, profile_id=profile_id)


def get_confidence_consensus(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_consensus(source_path=source_path, profile_id=profile_id)


def get_confidence_intelligence_v16(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_intelligence_v16(source_path=source_path, profile_id=profile_id)


def run_confidence_v16_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_confidence_v16_cycle(source_path=source_path, profile_id=profile_id)


def close_cace_v16_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
