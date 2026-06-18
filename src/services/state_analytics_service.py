"""State Analytics service layer."""
from __future__ import annotations

from typing import Any

from src.state_analytics.state_analytics_engine import StateAnalyticsEngine
from src.state_analytics.state_history_repository import StateHistoryRepository
from src.state_analytics.state_snapshot_writer import StateSnapshotWriter


class StateAnalyticsService:
    def __init__(
        self,
        *,
        analytics: StateAnalyticsEngine | None = None,
        writer: StateSnapshotWriter | None = None,
        owns_connections: bool = False,
    ) -> None:
        self._owns = owns_connections or analytics is None
        self._analytics = analytics or StateAnalyticsEngine(owns_connection=self._owns)
        if writer is not None:
            self._writer = writer
        else:
            history_repo = StateHistoryRepository(self._analytics._repo._db, owns_connection=False)
            self._writer = StateSnapshotWriter(repo=history_repo, owns_connection=False)

    def close(self) -> None:
        if self._owns:
            self._analytics.close()

    def get_account_state_summary(
        self,
        *,
        current_state: str = "unknown",
        current_profile: str = "unknown",
    ) -> dict[str, Any]:
        return self._analytics.build_summary(
            current_state=current_state,
            current_profile=current_profile,
        )

    @property
    def writer(self) -> StateSnapshotWriter:
        return self._writer

    @property
    def analytics(self) -> StateAnalyticsEngine:
        return self._analytics
