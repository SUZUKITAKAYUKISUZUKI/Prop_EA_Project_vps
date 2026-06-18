"""Challenge / Funded Auto Switch Engine."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from src.account_state_engine.account_state_engine import (
    AccountState,
    AccountStateEngine,
    AccountStateInput,
)
from src.auto_switch_engine.switch_repository import AutoSwitchRepository
from src.objective_optimizer.objective_profiles import recommended_objective_label
from src.profile_manager.resolver import resolve_profile_from_state, runtime_flags_for_profile
from src.services.profile_service import ProfileService
from src.state_analytics.state_snapshot_writer import StateSnapshotWriter


@dataclass(frozen=True)
class SwitchResult:
    switched: bool
    previous_profile: str | None
    new_profile: str
    account_state: str
    reason: str
    timestamp: str
    state_snapshot: dict[str, Any]
    recommended_objective: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "switched": self.switched,
            "previous_profile": self.previous_profile,
            "new_profile": self.new_profile,
            "account_state": self.account_state,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "state_snapshot": self.state_snapshot,
            "recommended_objective": self.recommended_objective,
        }


class AutoSwitchEngine:
    """Map account state → objective profile → active profile with event logging."""

    def __init__(
        self,
        *,
        profile_service: ProfileService | None = None,
        switch_repo: AutoSwitchRepository | None = None,
        state_engine: AccountStateEngine | None = None,
        snapshot_writer: StateSnapshotWriter | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections or profile_service is None
        self._profiles = profile_service or ProfileService()
        self._repo = switch_repo or AutoSwitchRepository(owns_connection=self._owns)
        self._state_engine = state_engine or AccountStateEngine()
        if snapshot_writer is not None:
            self._snapshots = snapshot_writer
        else:
            from src.state_analytics.state_history_repository import StateHistoryRepository

            history_repo = StateHistoryRepository(self._repo._db, owns_connection=False)
            self._snapshots = StateSnapshotWriter(repo=history_repo, owns_connection=False)

    def close(self) -> None:
        if self._owns:
            self._profiles.close()
            self._repo.close()

    @staticmethod
    def _apply_runtime_flags(flags: dict[str, Any]) -> None:
        os.environ["PORTFOLIO_ALLOCATION_ENABLED"] = "1" if flags["allocation_enabled"] else "0"
        os.environ["PORTFOLIO_WEIGHT_MODE"] = str(flags["portfolio_weight_mode"])
        os.environ["PORTFOLIO_ALLOCATION_SOURCE"] = str(flags["allocation_source"])
        os.environ["PROFILE_BAYES_THRESHOLD"] = str(flags["bayes_threshold"])
        os.environ["SIZING_MODEL"] = str(flags["sizing_model"])
        os.environ["PORTFOLIO_RISK_MULTIPLIER"] = str(flags["risk_multiplier"])
        os.environ["PROFIT_CUSHION_ENABLED"] = "1" if flags.get("profit_cushion", 0) else "0"
        if flags.get("recommended_objective"):
            os.environ["RECOMMENDED_OBJECTIVE"] = str(flags["recommended_objective"])

    def evaluate_state(self, inputs: AccountStateInput) -> dict[str, Any]:
        result = self._state_engine.evaluate(inputs)
        state = result.state
        return {
            **result.to_dict(),
            "target_profile": resolve_profile_from_state(state),
            "recommended_objective": recommended_objective_label(state),
        }

    def _persist_state_snapshot(
        self,
        inputs: AccountStateInput,
        state_result: Any,
        profile: str,
        *,
        previous_profile: str | None,
        source: str = "auto_switch",
    ) -> dict[str, Any]:
        last = self._snapshots._repo.get_latest()
        prev_state = str(last.get("state", "")).lower() if last else ""
        state_changed = prev_state != state_result.state.value
        profile_changed = (previous_profile or "") != profile
        return self._snapshots.record_if_due(
            state=state_result.state.value,
            profile=profile,
            equity=float(inputs.current_balance),
            balance=float(inputs.current_balance),
            dd_pct=float(state_result.dd_pct),
            risk_budget_remaining=float(state_result.risk_budget_remaining),
            challenge_progress=float(state_result.progress_pct),
            source=source,
            state_changed=state_changed,
            profile_changed=profile_changed,
        )

    def record_trade_completion_snapshot(
        self,
        inputs: AccountStateInput,
        *,
        profile: str | None = None,
    ) -> dict[str, Any]:
        state_result = self._state_engine.evaluate(inputs)
        target_profile = profile or resolve_profile_from_state(state_result.state)
        return self._snapshots.record_trade_completion(
            state=state_result.state.value,
            profile=target_profile,
            equity=float(inputs.current_balance),
            balance=float(inputs.current_balance),
            dd_pct=float(state_result.dd_pct),
            risk_budget_remaining=float(state_result.risk_budget_remaining),
            challenge_progress=float(state_result.progress_pct),
        )

    def evaluate_and_switch(
        self,
        inputs: AccountStateInput,
        *,
        force: bool = False,
    ) -> SwitchResult:
        state_result = self._state_engine.evaluate(inputs)
        state = state_result.state
        target_profile = resolve_profile_from_state(state)

        try:
            current = self._profiles.load_active_profile().profile_id
        except RuntimeError:
            current = None

        reason = self._build_reason(state, state_result)
        from src.database.db_manager import utc_now_iso

        timestamp = utc_now_iso()

        if not force and current == target_profile:
            self._persist_state_snapshot(
                inputs,
                state_result,
                target_profile,
                previous_profile=current,
            )
            return SwitchResult(
                switched=False,
                previous_profile=current,
                new_profile=target_profile,
                account_state=state.value,
                reason="already_active",
                timestamp=timestamp,
                state_snapshot=state_result.to_dict(),
                recommended_objective=recommended_objective_label(state),
            )

        if current != target_profile or force:
            record = self._profiles.get_profile(target_profile)
            flags = runtime_flags_for_profile(record)
            self._profiles.apply_profile(target_profile)
            self._apply_runtime_flags(flags)

        self._repo.log_switch(
            old_profile=current,
            new_profile=target_profile,
            account_state=state.value,
            reason=reason,
            equity=float(inputs.current_balance),
            dd=float(inputs.current_dd),
        )

        self._persist_state_snapshot(
            inputs,
            state_result,
            target_profile,
            previous_profile=current,
            source="profile_switch" if current != target_profile else "auto_switch",
        )

        return SwitchResult(
            switched=True,
            previous_profile=current,
            new_profile=target_profile,
            account_state=state.value,
            reason=reason,
            timestamp=timestamp,
            state_snapshot=state_result.to_dict(),
            recommended_objective=recommended_objective_label(state),
        )

    @staticmethod
    def _build_reason(state: AccountState, snapshot: Any) -> str:
        if state == AccountState.FUNDED:
            return "challenge_passed"
        if state == AccountState.RECOVERY:
            return f"dd_above_70pct_limit ({snapshot.dd_pct}% used)"
        if state == AccountState.LIVE:
            return "live_account_type"
        return "challenge_mode_default"

    def dashboard_snapshot(self, inputs: AccountStateInput) -> dict[str, Any]:
        evaluation = self.evaluate_state(inputs)
        try:
            active = self._profiles.load_active_profile()
            active_id = active.profile_id
            flags = runtime_flags_for_profile(self._profiles.get_profile(active_id))
        except (RuntimeError, KeyError):
            active_id = evaluation["target_profile"]
            flags = {}

        return {
            "current_state": evaluation["state"],
            "current_profile": active_id,
            "target_profile": evaluation["target_profile"],
            "progress_pct": evaluation["progress_pct"],
            "dd_pct": evaluation["dd_pct"],
            "risk_budget_remaining": evaluation["risk_budget_remaining"],
            "recommended_objective": evaluation["recommended_objective"],
            "allocation_enabled": flags.get("allocation_enabled"),
            "risk_multiplier": flags.get("risk_multiplier"),
            "switch_history": self._repo.recent_switches(limit=10),
        }
