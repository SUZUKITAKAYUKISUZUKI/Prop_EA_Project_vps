"""Dashboard API for live trade_events (Dropbox → SQLite pipeline)."""
from __future__ import annotations

from src.repositories.trade_event_repository import TradeEventRepository
from src.services.trade_event_analytics import TradeEventAnalytics


def get_trade_event_summary() -> dict:
    analytics = TradeEventAnalytics()
    summary = analytics.summary()
    return {
        "total_events": summary["total_events"],
        "trade_frequency_per_hour": round(summary["trade_frequency_per_hour"], 4),
        "pet_interventions": summary["pet_interventions"],
        "sentinel_interventions": summary["sentinel_interventions"],
        "daily_open_count": summary["daily_open_count"],
        "daily_close_count": summary["daily_close_count"],
        "events_per_day": summary["events_per_day"],
    }


def get_recent_events(limit: int = 50) -> list[dict]:
    repo = TradeEventRepository()
    rows = repo.get_recent_events(limit=limit)
    repo.close()
    return [
        {
            "event_id": row["event_id"],
            "timestamp": row["timestamp"],
            "event_type": row["event_type"],
            "trade_id": row.get("trade_id"),
            "strategy": row.get("strategy"),
            "symbol": row.get("symbol"),
        }
        for row in rows
    ]


def get_strategy_activity() -> dict[str, int]:
    return TradeEventAnalytics().strategy_activity()


def get_event_timeline(*, limit: int = 200) -> list[dict]:
    repo = TradeEventRepository()
    rows = repo.get_recent_events(limit=limit)
    repo.close()
    timeline = []
    for row in reversed(rows):
        timeline.append(
            {
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "trade_id": row.get("trade_id"),
                "strategy": row.get("strategy"),
                "symbol": row.get("symbol"),
                "detail": {k: v for k, v in row.items() if k not in {"event_id", "timestamp", "event_type", "trade_id", "strategy", "symbol", "imported_at", "payload_json"}},
            }
        )
    return timeline


def get_feature_summary(*, limit: int = 100) -> dict:
    repo = TradeEventRepository()
    snapshots = repo.get_feature_snapshots(limit=limit)
    sqlite_features = repo.count_features()
    repo.close()
    strategies: dict[str, int] = {}
    for row in snapshots:
        strategy = str(row.get("strategy") or "UNKNOWN")
        strategies[strategy] = strategies.get(strategy, 0) + 1
    return {
        "feature_snapshots": len(snapshots),
        "sqlite_features": sqlite_features,
        "strategies": strategies,
    }


def get_recent_features(*, limit: int = 50) -> list[dict]:
    repo = TradeEventRepository()
    rows = repo.get_feature_snapshots(limit=limit)
    repo.close()
    out: list[dict] = []
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        out.append(
            {
                "event_id": row.get("event_id"),
                "timestamp": row.get("timestamp"),
                "trade_id": row.get("trade_id"),
                "strategy": row.get("strategy"),
                "symbol": row.get("symbol"),
                "bayes_probability": features.get("bayes_probability"),
                "candidate_score": features.get("candidate_score"),
                "atr_ratio": features.get("atr_ratio"),
                "session": features.get("session"),
                "decision_source": features.get("decision_source"),
            }
        )
    return out
