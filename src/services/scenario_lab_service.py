"""PDTS service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.api.adaptive_allocation_api import get_adaptive_allocation
from src.digital_twin.scenario_builder import SCENARIO_BASELINE, SCENARIO_RECOMMENDED
from src.digital_twin.twin_engine import TwinEngine
from src.services.profile_service import ProfileService
from src.services.risk_attribution_v2_service import RiskAttributionV2Service


class ScenarioLabService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._engine = TwinEngine(owns_connections=owns_connections)
        self._profiles = ProfileService()
        self._trades = RiskAttributionV2Service(owns_connections=False)

    def close(self) -> None:
        if self._owns:
            self._engine.close()
        self._profiles.close()
        self._trades.close()

    def run_scenario(
        self,
        *,
        profile_id: str | None = None,
        scenario: str = SCENARIO_RECOMMENDED,
        custom_allocation: dict[str, float] | None = None,
        source_path: str | Path | None = None,
        persist: bool = True,
        mc_fast: bool = True,
        created_by: str = "scenario_lab",
    ) -> dict[str, Any]:
        ctx = self._profiles.load_active_profile()
        pid = profile_id or ctx.profile_id
        if pid != ctx.profile_id:
            record = self._profiles.get_profile(pid)
            ctx = type(ctx).from_record(record)

        paae = get_adaptive_allocation(source_path=source_path, profile_id=pid)
        trades = self._trades.load_trades(source_path=source_path)

        return self._engine.run_scenario(
            scenario_name=scenario,
            profile_ctx=ctx,
            trades=trades,
            paae_report=paae,
            custom_allocation=custom_allocation,
            persist=persist,
            created_by=created_by,
            mc_fast=mc_fast,
        )

    def compare_all(
        self,
        *,
        profile_id: str | None = None,
        source_path: str | Path | None = None,
        persist: bool = False,
        mc_fast: bool = True,
    ) -> dict[str, Any]:
        ctx = self._profiles.load_active_profile()
        pid = profile_id or ctx.profile_id
        if pid != ctx.profile_id:
            record = self._profiles.get_profile(pid)
            ctx = type(ctx).from_record(record)

        paae = get_adaptive_allocation(source_path=source_path, profile_id=pid)
        trades = self._trades.load_trades(source_path=source_path)

        return self._engine.compare_scenarios(
            profile_ctx=ctx,
            trades=trades,
            paae_report=paae,
            persist=persist,
            mc_fast=mc_fast,
        )

    def get_scenario_lab(
        self,
        *,
        profile_id: str | None = None,
        source_path: str | Path | None = None,
        mc_fast: bool = True,
    ) -> dict[str, Any]:
        comparison = self.compare_all(
            profile_id=profile_id,
            source_path=source_path,
            persist=False,
            mc_fast=mc_fast,
        )
        baseline = comparison.get("baseline") or {}
        recommended = comparison.get("recommended") or {}
        return {
            "profile_id": comparison.get("profile_id"),
            "account_state": comparison.get("account_state"),
            "scenario_comparison": {
                "current": baseline,
                "recommended": recommended,
            },
            "pass_probability": [
                {"scenario": r.get("scenario_label"), "pass_rate": r.get("pass_rate")}
                for r in comparison.get("scenarios") or []
            ],
            "expected_dd": [
                {"scenario": r.get("scenario_label"), "max_dd": r.get("max_dd")}
                for r in comparison.get("scenarios") or []
            ],
            "expected_profit": [
                {"scenario": r.get("scenario_label"), "total_r": r.get("total_r")}
                for r in comparison.get("scenarios") or []
            ],
            "recommendation_ranking": comparison.get("ranking") or [],
            "allocation_impact": comparison.get("allocation_impact") or [],
        }
