"""Persist account state snapshots to SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.state_analytics.state_history_repository import StateHistoryRepository

DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 3600


class StateSnapshotWriter:
    """Write state snapshots to account_state_history with optional throttling."""

    def __init__(
        self,
        repo: StateHistoryRepository | None = None,
        *,
        owns_connection: bool = False,
        interval_seconds: int = DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
    ) -> None:
        self._repo = repo or StateHistoryRepository(owns_connection=owns_connection)
        self._owns = owns_connection or repo is None
        self._interval_seconds = interval_seconds

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def record_snapshot(
        self,
        *,
        state: str,
        profile: str,
        equity: float | None = None,
        balance: float | None = None,
        dd_pct: float | None = None,
        risk_budget_remaining: float | None = None,
        challenge_progress: float | None = None,
        source: str = "auto_switch",
        force: bool = False,
    ) -> dict[str, Any]:
        """Insert a snapshot row. Returns metadata about the write."""
        if not force:
            last = self._repo.get_latest()
            if last is not None:
                same_state = str(last.get("state", "")).lower() == str(state).lower()
                same_profile = str(last.get("profile", "")) == str(profile)
                if same_state and same_profile and not self._interval_elapsed(last["timestamp"]):
                    return {"recorded": False, "reason": "throttled", "snapshot_id": last.get("id")}

        snapshot_id = self._repo.insert_snapshot(
            state=str(state).lower(),
            profile=str(profile),
            equity=equity,
            balance=balance if balance is not None else equity,
            drawdown_pct=dd_pct,
            risk_budget_remaining=risk_budget_remaining,
            challenge_progress=challenge_progress,
            source=source,
        )
        return {"recorded": True, "reason": "inserted", "snapshot_id": snapshot_id}

    def record_if_due(
        self,
        *,
        state: str,
        profile: str,
        equity: float | None = None,
        balance: float | None = None,
        dd_pct: float | None = None,
        risk_budget_remaining: float | None = None,
        challenge_progress: float | None = None,
        source: str = "auto_switch",
        state_changed: bool = False,
        profile_changed: bool = False,
    ) -> dict[str, Any]:
        force = state_changed or profile_changed
        return self.record_snapshot(
            state=state,
            profile=profile,
            equity=equity,
            balance=balance,
            dd_pct=dd_pct,
            risk_budget_remaining=risk_budget_remaining,
            challenge_progress=challenge_progress,
            source=source,
            force=force,
        )

    def record_trade_completion(
        self,
        *,
        state: str,
        profile: str,
        equity: float | None = None,
        balance: float | None = None,
        dd_pct: float | None = None,
        risk_budget_remaining: float | None = None,
        challenge_progress: float | None = None,
    ) -> dict[str, Any]:
        return self.record_snapshot(
            state=state,
            profile=profile,
            equity=equity,
            balance=balance,
            dd_pct=dd_pct,
            risk_budget_remaining=risk_budget_remaining,
            challenge_progress=challenge_progress,
            source="trade_completion",
            force=True,
        )

    def _interval_elapsed(self, timestamp: str) -> bool:
        try:
            normalized = timestamp.replace("Z", "+00:00")
            last_dt = datetime.fromisoformat(normalized)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - last_dt).total_seconds() >= self._interval_seconds
        except ValueError:
            return True
