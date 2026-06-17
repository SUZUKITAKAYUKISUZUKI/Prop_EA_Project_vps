"""Analytics over imported live trade_events."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from src.repositories.trade_event_repository import TradeEventRepository


class TradeEventAnalytics:
    def __init__(self, repo: TradeEventRepository | None = None) -> None:
        self.repo = repo or TradeEventRepository()

    def _day_key(self, timestamp: str) -> str:
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            return timestamp[:10]

    def events_per_day(self, *, limit: int = 50000) -> dict[str, int]:
        rows = self.repo.get_recent_events(limit=limit)
        counter: Counter[str] = Counter()
        for row in rows:
            counter[self._day_key(str(row["timestamp"]))] += 1
        return dict(sorted(counter.items()))

    def strategy_activity(self, *, limit: int = 50000) -> dict[str, int]:
        rows = self.repo.get_recent_events(limit=limit)
        counter: Counter[str] = Counter()
        for row in rows:
            strategy = row.get("strategy") or "UNKNOWN"
            counter[str(strategy)] += 1
        return dict(counter.most_common())

    def symbol_activity(self, *, limit: int = 50000) -> dict[str, int]:
        rows = self.repo.get_recent_events(limit=limit)
        counter: Counter[str] = Counter()
        for row in rows:
            symbol = row.get("symbol") or "UNKNOWN"
            counter[str(symbol)] += 1
        return dict(counter.most_common())

    def trade_frequency(self, *, limit: int = 50000) -> float:
        rows = self.repo.get_recent_events(limit=limit)
        if len(rows) < 2:
            return 0.0
        timestamps = sorted(
            datetime.fromisoformat(str(r["timestamp"]).replace("Z", "+00:00")) for r in rows
        )
        span_hours = max((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0, 1e-9)
        return float(len(rows) / span_hours)

    def daily_open_count(self) -> dict[str, int]:
        return self._count_by_day("TRADE_OPEN")

    def daily_close_count(self) -> dict[str, int]:
        return self._count_by_day("TRADE_CLOSE")

    def pet_interventions(self) -> int:
        return len(self.repo.get_events_by_type("PET_EXIT", limit=100000))

    def sentinel_interventions(self) -> int:
        return len(self.repo.get_events_by_type("SENTINEL_BLOCK", limit=100000))

    def _count_by_day(self, event_type: str) -> dict[str, int]:
        rows = self.repo.get_events_by_type(event_type, limit=100000)
        counter: Counter[str] = Counter()
        for row in rows:
            counter[self._day_key(str(row["timestamp"]))] += 1
        return dict(sorted(counter.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "total_events": self.repo.count_events(),
            "events_per_day": self.events_per_day(),
            "strategy_activity": self.strategy_activity(),
            "symbol_activity": self.symbol_activity(),
            "trade_frequency_per_hour": self.trade_frequency(),
            "daily_open_count": self.daily_open_count(),
            "daily_close_count": self.daily_close_count(),
            "pet_interventions": self.pet_interventions(),
            "sentinel_interventions": self.sentinel_interventions(),
            "feature_snapshots": len(self.repo.get_events_by_type("FEATURE_SNAPSHOT", limit=100000)),
            "sqlite_features": self.repo.count_features(),
        }
