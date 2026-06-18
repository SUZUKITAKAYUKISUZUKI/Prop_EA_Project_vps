"""Dashboard API — State Analytics Engine v1."""
from __future__ import annotations

from typing import Any

from src.services.state_analytics_service import StateAnalyticsService

_svc: StateAnalyticsService | None = None


def _service() -> StateAnalyticsService:
    global _svc
    if _svc is None:
        _svc = StateAnalyticsService(owns_connections=True)
    return _svc


def get_account_state_summary(
    dashboard_state: dict[str, Any] | None = None,
    *,
    current_state: str | None = None,
    current_profile: str | None = None,
) -> dict[str, Any]:
    state = current_state
    profile = current_profile
    if dashboard_state is not None:
        from src.api.auto_switch_api import get_account_state_snapshot

        snap = get_account_state_snapshot(dashboard_state)
        state = snap.get("current_state") or state
        profile = snap.get("current_profile") or profile
    return _service().get_account_state_summary(
        current_state=str(state or "unknown"),
        current_profile=str(profile or "unknown"),
    )


def close_state_analytics_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
