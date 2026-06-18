"""Public API for Portfolio OS RC1 production hardening."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.production_hardening_service import ProductionHardeningService

_svc: ProductionHardeningService | None = None


def _service() -> ProductionHardeningService:
    global _svc
    if _svc is None:
        _svc = ProductionHardeningService(owns_connections=True)
    return _svc


def get_production_readiness(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_production_readiness(source_path=source_path, profile_id=profile_id)


def get_resilience_score(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_resilience_score(source_path=source_path, profile_id=profile_id)


def get_validation_results(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_validation_results(source_path=source_path, profile_id=profile_id)


def get_benchmark_results(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_benchmark_results(source_path=source_path, profile_id=profile_id)


def run_production_validation(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_production_validation(source_path=source_path, profile_id=profile_id)


def close_production_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
