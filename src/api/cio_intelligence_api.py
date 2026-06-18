"""Public API for CIO Intelligence Layer v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.cio_intelligence_service import CioIntelligenceService

_svc: CioIntelligenceService | None = None


def _service() -> CioIntelligenceService:
    global _svc
    if _svc is None:
        _svc = CioIntelligenceService(owns_connections=True)
    return _svc


def get_investment_state(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[str]:
    return _service().get_investment_state(source_path=source_path, profile_id=profile_id)


def get_executive_score(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_executive_score(source_path=source_path, profile_id=profile_id)


def get_opportunity_report(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_opportunity_report(source_path=source_path, profile_id=profile_id)


def get_risk_report(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_risk_report(source_path=source_path, profile_id=profile_id)


def get_capital_efficiency(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_capital_efficiency(source_path=source_path, profile_id=profile_id)


def get_cio_intelligence(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_cio_intelligence(source_path=source_path, profile_id=profile_id)


def run_cio_intelligence_cycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().run_cio_intelligence_cycle(source_path=source_path, profile_id=profile_id)


def close_cio_intelligence_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
