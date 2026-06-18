"""AI Governor Engine service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ai_governor.engine import AiGovernorEngine
from src.api.adaptive_allocation_api import get_adaptive_allocation
from src.api.lifecycle_api import get_strategy_lifecycle
from src.api.risk_attribution_v2_api import get_risk_attribution_v2
from src.api.scenario_lab_api import get_scenario_lab
from src.api.state_analytics_api import get_account_state_summary
from src.services.profile_service import ProfileService


class AiGovernorService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._engine = AiGovernorEngine(owns_connections=owns_connections)
        self._profiles = ProfileService()

    def close(self) -> None:
        if self._owns:
            self._engine.close()
        self._profiles.close()

    def _load_upstream(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
        ctx = self._profiles.load_active_profile()
        pid = profile_id or ctx.profile_id
        state = str(ctx.settings.get("account_state") or "unknown")
        prae_v2 = get_risk_attribution_v2(source_path=source_path, profile_id=pid)
        state_summary = get_account_state_summary(current_state=state, current_profile=pid)
        paae = get_adaptive_allocation(source_path=source_path, profile_id=pid)
        pdts = get_scenario_lab(profile_id=pid, source_path=source_path, mc_fast=True)
        slm = get_strategy_lifecycle(source_path=source_path, profile_id=pid)
        return ctx, ctx.to_dict(), prae_v2, state_summary, paae, pdts, slm, pid

    def get_governor_status(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        report = self.get_governor_report(source_path=source_path, profile_id=profile_id, persist=False)
        if report:
            return self._engine._reporter.status_payload(report)
        return self._engine.get_governor_status()

    def get_governor_report(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        ctx, profile_dict, prae_v2, state_summary, paae, pdts, slm, _ = self._load_upstream(
            source_path=source_path,
            profile_id=profile_id,
        )
        context = self._engine.load_context(
            profile_ctx=profile_dict,
            prae_v2=prae_v2,
            state_analytics=state_summary,
            paae=paae,
            pdts=pdts,
            slm=slm,
        )
        return self._engine.run_governor_cycle(
            context,
            persist=persist,
            created_by="age_dashboard",
        )

    def get_governor_recommendations(
        self,
        *,
        status: str = "OPEN",
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.get_governor_report(source_path=source_path, profile_id=profile_id, persist=True)
        return self._engine.get_governor_recommendations(status=status)

    def run_governor_cycle(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        return self.get_governor_report(
            source_path=source_path,
            profile_id=profile_id,
            persist=True,
        )

    def get_governor_decision_history(
        self,
        *,
        profile_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._engine.get_decision_history(profile_id=profile_id, limit=limit)
