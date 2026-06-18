"""API for AGE v3 Predictive Governor."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.predictive_governor_service import PredictiveGovernorService

_svc: PredictiveGovernorService | None = None


def _service() -> PredictiveGovernorService:
    global _svc
    if _svc is None:
        _svc = PredictiveGovernorService(owns_connections=True)
    return _svc


def get_predictive_governor_report(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_predictive_governor_report(
        source_path=source_path,
        profile_id=profile_id,
        persist=False,
    )


def get_health_forecast(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_health_forecast(source_path=source_path, profile_id=profile_id)


def get_recovery_forecast(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_recovery_forecast(source_path=source_path, profile_id=profile_id)


def get_profile_transition_forecast(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_profile_transition_forecast(source_path=source_path, profile_id=profile_id)


def get_strategy_forecast(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_strategy_forecast(source_path=source_path, profile_id=profile_id)


def get_predictive_alerts(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _service().get_predictive_alerts(
        source_path=source_path,
        profile_id=profile_id,
        limit=limit,
    )


def run_predictive_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_predictive_cycle(source_path=source_path, profile_id=profile_id)


def close_age_v3_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
