"""Dashboard API for Strategy Lifecycle Manager."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.strategy_lifecycle_service import StrategyLifecycleService

_svc: StrategyLifecycleService | None = None


def _service() -> StrategyLifecycleService:
    global _svc
    if _svc is None:
        _svc = StrategyLifecycleService(owns_connections=True)
    return _svc


def get_strategy_lifecycle(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_strategy_lifecycle(source_path=source_path, profile_id=profile_id)


def get_portfolio_fit_score(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_portfolio_fit_score(source_path=source_path, profile_id=profile_id)


def get_strategy_portfolio_fit(
    strategy_id: str,
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_strategy_portfolio_fit(
        strategy_id,
        source_path=source_path,
        profile_id=profile_id,
    )


def get_portfolio_fit_ranking(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_portfolio_fit_ranking(source_path=source_path, profile_id=profile_id)


def evaluate_strategy(
    strategy_id: str,
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().evaluate_strategy(
        strategy_id,
        source_path=source_path,
        profile_id=profile_id,
    )


def promote_strategy(
    strategy_id: str,
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    return _service().promote_strategy(
        strategy_id,
        source_path=source_path,
        profile_id=profile_id,
        force=force,
    )


def retire_strategy(strategy_id: str, *, reason: str = "manual_retire") -> dict[str, Any]:
    return _service().retire_strategy(strategy_id, reason=reason)


def run_weekly_lifecycle_evaluation(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    return _service().run_weekly_evaluation(
        source_path=source_path,
        profile_id=profile_id,
        force=force,
    )


def get_strategy_genealogy(strategy_id: str | None = None) -> dict[str, Any]:
    return _service().get_strategy_genealogy(strategy_id)


def get_strategy_explanation(
    strategy_id: str,
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_strategy_explanation(
        strategy_id,
        source_path=source_path,
        profile_id=profile_id,
    )


def get_core_strategies() -> list[dict[str, Any]]:
    return _service().get_core_strategies()


def get_lifecycle_decision_report(
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    return _service().get_lifecycle_decision_report(
        source_path=source_path,
        profile_id=profile_id,
    )


def get_strategy_governance_context(
    strategy_id: str,
    *,
    source_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    return _service().get_strategy_governance_context(
        strategy_id,
        source_path=source_path,
        profile_id=profile_id,
    )


def close_lifecycle_api() -> None:
    global _svc
    if _svc is not None:
        _svc.close()
        _svc = None
