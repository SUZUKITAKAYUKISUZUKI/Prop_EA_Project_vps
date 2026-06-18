"""Dashboard API for Portfolio Digital Twin & Scenario Simulator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.scenario_lab_service import ScenarioLabService

_svc: ScenarioLabService | None = None


def _service() -> ScenarioLabService:
    global _svc
    if _svc is None:
        _svc = ScenarioLabService(owns_connections=True)
    return _svc


def run_scenario(
    profile_id: str | None = None,
    scenario: str = "recommended",
    *,
    custom_allocation: dict[str, float] | None = None,
    source_path: str | Path | None = None,
    persist: bool = True,
    mc_fast: bool = True,
) -> dict[str, Any]:
    return _service().run_scenario(
        profile_id=profile_id,
        scenario=scenario,
        custom_allocation=custom_allocation,
        source_path=source_path,
        persist=persist,
        mc_fast=mc_fast,
    )


def compare_scenarios(
    profile_id: str | None = None,
    *,
    source_path: str | Path | None = None,
    persist: bool = False,
    mc_fast: bool = True,
) -> dict[str, Any]:
    return _service().compare_all(
        profile_id=profile_id,
        source_path=source_path,
        persist=persist,
        mc_fast=mc_fast,
    )


def get_scenario_lab(
    profile_id: str | None = None,
    *,
    source_path: str | Path | None = None,
    mc_fast: bool = True,
) -> dict[str, Any]:
    return _service().get_scenario_lab(
        profile_id=profile_id,
        source_path=source_path,
        mc_fast=mc_fast,
    )


def close_scenario_lab_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
