"""Dashboard API for Account State + Auto Switch Engine."""
from __future__ import annotations

from typing import Any

from audit.risk_manager import STARTING_EQUITY
from core.pass_probability import AccountSnapshot, ChallengeState
from src.account_state_engine.account_state_engine import AccountStateInput
from src.auto_switch_engine.auto_switch_engine import AutoSwitchEngine

_engine_instance: AutoSwitchEngine | None = None


def _get_engine() -> AutoSwitchEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AutoSwitchEngine(owns_connections=True)
    return _engine_instance


def _inputs_from_dashboard_state(state: dict[str, Any]) -> AccountStateInput:
    account = state.get("account") or {}
    challenge = state.get("challenge") or {}
    equity = float(account.get("equity") or STARTING_EQUITY)
    starting = float(account.get("phase_start_equity") or STARTING_EQUITY)
    target_profit_pct = float(state.get("target_profit") or 8.0)
    target_balance = starting * (1.0 + target_profit_pct / 100.0)
    max_dd = float(state.get("total_dd_limit") or 8.5)
    current_dd = float(challenge.get("total_dd_used_percent") or 0.0)
    progress = float(challenge.get("profit_progress_percent") or 0.0)
    return AccountStateInput(
        current_balance=equity,
        initial_balance=starting,
        target_balance=target_balance,
        max_total_dd=max_dd,
        current_dd=current_dd,
        account_type=str(state.get("account_type") or "prop"),
        challenge_passed=progress >= target_profit_pct,
    )


def get_account_state_snapshot(dashboard_state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = dashboard_state or {}
    return _get_engine().dashboard_snapshot(_inputs_from_dashboard_state(state))


def run_auto_switch(dashboard_state: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    result = _get_engine().evaluate_and_switch(
        _inputs_from_dashboard_state(dashboard_state),
        force=force,
    )
    snapshot = _get_engine().dashboard_snapshot(_inputs_from_dashboard_state(dashboard_state))
    return {"switch": result.to_dict(), "snapshot": snapshot}


def close_auto_switch_api() -> None:
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.close()
        _engine_instance = None
