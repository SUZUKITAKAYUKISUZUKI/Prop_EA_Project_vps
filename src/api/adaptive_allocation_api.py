"""Dashboard API for Portfolio Adaptive Allocation Engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.adaptive_allocation_service import AdaptiveAllocationService

_svc: AdaptiveAllocationService | None = None


def _service() -> AdaptiveAllocationService:
    global _svc
    if _svc is None:
        _svc = AdaptiveAllocationService(owns_connections=True)
    return _svc


def get_adaptive_allocation(
    source_path: str | Path | None = None,
    *,
    profile_id: str | None = None,
    rebalance: bool = False,
    force: bool = False,
    trigger: str | None = None,
) -> dict[str, Any]:
    return _service().get_adaptive_allocation(
        source_path=source_path,
        profile_id=profile_id,
        rebalance=rebalance,
        force=force,
        trigger=trigger,
    )


def close_adaptive_allocation_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
