"""API for CACE v1.7 Confidence Calibration Intelligence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.cace_v17_service import CaceV17Service

_svc: CaceV17Service | None = None


def _service() -> CaceV17Service:
    global _svc
    if _svc is None:
        _svc = CaceV17Service(owns_connections=True)
    return _svc


def get_confidence_calibration(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_calibration(source_path=source_path, profile_id=profile_id)


def get_decision_accuracy(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_decision_accuracy(source_path=source_path, profile_id=profile_id)


def get_confidence_reliability(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_reliability(source_path=source_path, profile_id=profile_id)


def get_recommendation_accuracy(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_recommendation_accuracy(source_path=source_path, profile_id=profile_id)


def get_confidence_learning(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_learning(source_path=source_path, profile_id=profile_id)


def get_confidence_intelligence_v17(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_confidence_intelligence_v17(source_path=source_path, profile_id=profile_id)


def run_calibration_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_calibration_cycle(source_path=source_path, profile_id=profile_id)


def close_cace_v17_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
