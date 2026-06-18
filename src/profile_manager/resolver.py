"""Map account state to operating profile and runtime flags."""
from __future__ import annotations

from typing import Any

from src.account_state_engine.account_state_engine import AccountState

STATE_PROFILE_MAP: dict[AccountState, str] = {
    AccountState.CHALLENGE: "ChallengeAggressive",
    AccountState.FUNDED: "FundedBalanced",
    AccountState.RECOVERY: "RecoveryDefensive",
    AccountState.LIVE: "LiveCapitalGrowth",
}


def resolve_profile_from_state(state: AccountState | str) -> str:
    if isinstance(state, AccountState):
        return STATE_PROFILE_MAP[state]
    normalized = str(state).strip().lower()
    for account_state, profile_id in STATE_PROFILE_MAP.items():
        if account_state.value == normalized:
            return profile_id
    return STATE_PROFILE_MAP[AccountState.CHALLENGE]


def runtime_flags_for_profile(profile_record: dict[str, Any]) -> dict[str, Any]:
    """Extract auto-switch runtime flags from a hydrated profile record."""
    settings = profile_record.get("settings") or {}
    risk = profile_record.get("risk") or {}

    def _flag(key: str, default: str = "0") -> bool:
        return str(settings.get(key, default)).strip().lower() in {"1", "true", "yes", "on"}

    return {
        "allocation_enabled": _flag("allocation_enabled"),
        "portfolio_weight_mode": str(settings.get("portfolio_weight_mode", "equal")),
        "allocation_source": str(settings.get("allocation_source", "profile")),
        "bayes_threshold": float(settings.get("bayes_threshold", 0.8)),
        "sizing_model": str(settings.get("sizing_model", "A")),
        "profit_cushion": float(
            risk.get("profit_cushion_multiplier")
            or settings.get("profit_cushion")
            or 0.65
        ),
        "equity_trail_enabled": bool(risk.get("equity_trail_enabled")),
        "risk_multiplier": float(settings.get("risk_multiplier", 1.0)),
        "recommended_objective": str(settings.get("recommended_objective", "")),
        "profile_key": str(settings.get("profile_key", "challenge")),
    }
