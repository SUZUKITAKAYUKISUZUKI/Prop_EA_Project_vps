"""Public API for Meta Intelligence Engine v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.mie_service import MieService

_svc: MieService | None = None


def _service() -> MieService:
    global _svc
    if _svc is None:
        _svc = MieService(owns_connections=True)
    return _svc


def get_module_trust_scores(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_module_trust_scores(source_path=source_path, profile_id=profile_id)


def get_module_rankings(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_module_rankings(source_path=source_path, profile_id=profile_id)


def get_module_drift(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_module_drift(source_path=source_path, profile_id=profile_id)


def get_self_improvement_notes(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_self_improvement_notes(source_path=source_path, profile_id=profile_id)


def get_meta_intelligence(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_meta_intelligence(source_path=source_path, profile_id=profile_id)


def run_meta_intelligence_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_meta_intelligence_cycle(source_path=source_path, profile_id=profile_id)


def close_mie_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
