"""Dashboard API for Portfolio Risk Attribution Engine v2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.api.dashboard_api import DEFAULT_PORTFOLIO_SOURCE
from src.repositories.base import normalize_source_path
from src.services.risk_attribution_v2_service import RiskAttributionV2Service

_svc: RiskAttributionV2Service | None = None


def _service() -> RiskAttributionV2Service:
    global _svc
    if _svc is None:
        _svc = RiskAttributionV2Service(owns_connections=True)
    return _svc


def get_risk_attribution_v2(
    source_path: str | Path | None = None,
    *,
    run_id: int | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    report = _service().run_report(
        source_path=source_path,
        run_id=run_id,
        profile_id=profile_id,
    )
    return {
        "source": report.get("source") or normalize_source_path(source_path or DEFAULT_PORTFOLIO_SOURCE),
        "profile_id": report.get("profile_id"),
        "strategy_risk": report.get("strategy_risk", []),
        "weight_adjusted_contribution": report.get("weight_adjusted_contribution", []),
        "symbol_risk": report.get("symbol_risk", []),
        "recovery_analysis": report.get("recovery_analysis", []),
        "dd_attribution": report.get("dd_attribution", {}),
        "state_transition_risk": report.get("state_transition_risk", []),
        "health_report": report.get("health_report", {}),
        "top_risk_sources": report.get("top_risk_sources", []),
    }


def close_risk_attribution_v2_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
