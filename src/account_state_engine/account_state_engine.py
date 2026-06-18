"""Account State Engine — classify prop account phase for auto profile switching."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AccountState(str, Enum):
    CHALLENGE = "challenge"
    FUNDED = "funded"
    LIVE = "live"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class AccountStateInput:
    current_balance: float
    initial_balance: float
    target_balance: float
    max_total_dd: float
    current_dd: float
    account_type: str = "prop"
    challenge_passed: bool = False


@dataclass(frozen=True)
class AccountStateResult:
    state: AccountState
    progress_pct: float
    dd_pct: float
    risk_budget_remaining: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "progress_pct": self.progress_pct,
            "dd_pct": self.dd_pct,
            "risk_budget_remaining": self.risk_budget_remaining,
        }


class AccountStateEngine:
    """Determine account operating phase from balance, DD, and challenge status."""

    RECOVERY_DD_RATIO = 0.7

    def evaluate(self, inputs: AccountStateInput) -> AccountStateResult:
        dd_pct = round(max(0.0, float(inputs.current_dd)), 2)
        max_dd = max(float(inputs.max_total_dd), 1e-9)
        risk_budget_remaining = round(max(0.0, max_dd - dd_pct), 2)

        initial = float(inputs.initial_balance)
        current = float(inputs.current_balance)
        target = float(inputs.target_balance)
        profit_span = max(target - initial, 1e-9)
        progress_pct = round(max(0.0, (current - initial) / profit_span * 100.0), 1)

        state = self._resolve_state(inputs, dd_pct=dd_pct, max_dd=max_dd)
        return AccountStateResult(
            state=state,
            progress_pct=progress_pct,
            dd_pct=dd_pct,
            risk_budget_remaining=risk_budget_remaining,
        )

    def _resolve_state(
        self,
        inputs: AccountStateInput,
        *,
        dd_pct: float,
        max_dd: float,
    ) -> AccountState:
        if dd_pct > self.RECOVERY_DD_RATIO * max_dd:
            return AccountState.RECOVERY
        if inputs.challenge_passed:
            return AccountState.FUNDED
        if str(inputs.account_type).strip().lower() == "live":
            return AccountState.LIVE
        return AccountState.CHALLENGE
