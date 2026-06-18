"""PAAE service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.adaptive_allocation.engine import AdaptiveAllocationEngine
from src.api.risk_attribution_v2_api import get_risk_attribution_v2
from src.api.state_analytics_api import get_account_state_summary
from src.services.profile_service import ProfileService
from src.services.risk_attribution_v2_service import RiskAttributionV2Service


class AdaptiveAllocationService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._engine = AdaptiveAllocationEngine(owns_connections=owns_connections)
        self._prae = RiskAttributionV2Service(owns_connections=False)
        self._profiles = ProfileService()

    def close(self) -> None:
        if self._owns:
            self._engine.close()
            self._prae.close()
        self._profiles.close()

    def get_adaptive_allocation(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        rebalance: bool = False,
        force: bool = False,
        trigger: str | None = None,
    ) -> dict[str, Any]:
        ctx = self._profiles.load_active_profile()
        pid = profile_id or ctx.profile_id
        current = dict(ctx.strategy_allocations)

        prae_v2 = get_risk_attribution_v2(source_path=source_path, profile_id=pid)
        state_summary = get_account_state_summary(
            current_state=str(ctx.settings.get("account_state") or ""),
            current_profile=pid,
        )

        trades = self._prae.load_trades(source_path=source_path)

        if rebalance:
            return self._engine.rebalance(
                profile_id=pid,
                prae_v2=prae_v2,
                current_weights=current,
                profile_settings=ctx.settings,
                state_summary=state_summary,
                trades=trades,
                apply=True,
                force=force,
                trigger=trigger,
            )

        return self._engine.compute_allocation(
            prae_v2=prae_v2,
            current_weights=current,
            profile_id=pid,
            profile_settings=ctx.settings,
            state_summary=state_summary,
            trades=trades,
        )
