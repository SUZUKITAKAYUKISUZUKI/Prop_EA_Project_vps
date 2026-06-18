"""State Analytics Engine — historical state transition analysis."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.state_analytics.state_history_repository import StateHistoryRepository


def _parse_ts(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hours_between(start: str, end: str) -> float:
    return max(0.0, (_parse_ts(end) - _parse_ts(start)).total_seconds() / 3600.0)


def _days_between(start: str, end: str) -> float:
    return _hours_between(start, end) / 24.0


@dataclass(frozen=True)
class HealthScoreResult:
    score: float
    status: str
    recovery_frequency_penalty: float
    dd_penalty: float
    funded_duration_bonus: float
    challenge_success_bonus: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "health_score": round(self.score, 1),
            "status": self.status,
            "components": {
                "recovery_frequency_penalty": round(self.recovery_frequency_penalty, 2),
                "dd_penalty": round(self.dd_penalty, 2),
                "funded_duration_bonus": round(self.funded_duration_bonus, 2),
                "challenge_success_bonus": round(self.challenge_success_bonus, 2),
            },
        }


class StateAnalyticsEngine:
    """Analyze account_state_history for dwell time, transitions, and health."""

    TRANSITIONS = (
        ("challenge", "funded", "challenge_to_funded"),
        ("funded", "recovery", "funded_to_recovery"),
        ("recovery", "funded", "recovery_to_funded"),
        ("challenge", "recovery", "challenge_to_recovery"),
        ("live", "recovery", "live_to_recovery"),
    )

    def __init__(self, repo: StateHistoryRepository | None = None, *, owns_connection: bool = False) -> None:
        self._repo = repo or StateHistoryRepository(owns_connection=owns_connection)
        self._owns = owns_connection or repo is None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def _rows(self, *, limit: int = 10000) -> list[dict[str, Any]]:
        return self._repo.list_history(limit=limit)

    def total_days_per_state(self, *, limit: int = 10000) -> dict[str, float]:
        rows = self._rows(limit=limit)
        if not rows:
            return {s: 0.0 for s in ("challenge", "funded", "recovery", "live")}

        totals: dict[str, float] = {"challenge": 0.0, "funded": 0.0, "recovery": 0.0, "live": 0.0}
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        for idx, row in enumerate(rows):
            state = str(row["state"]).lower()
            start = str(row["timestamp"])
            end = str(rows[idx + 1]["timestamp"]) if idx + 1 < len(rows) else now_iso
            days = _days_between(start, end)
            totals[state] = totals.get(state, 0.0) + days
        return {k: round(v, 2) for k, v in totals.items()}

    def state_frequency(self, *, limit: int = 10000) -> dict[str, int]:
        rows = self._rows(limit=limit)
        counts: dict[str, int] = {"challenge": 0, "funded": 0, "recovery": 0, "live": 0}
        for row in rows:
            state = str(row["state"]).lower()
            counts[state] = counts.get(state, 0) + 1
        return counts

    def transition_counts(self, *, limit: int = 10000) -> dict[str, Any]:
        rows = self._rows(limit=limit)
        counts = {key: 0 for _, _, key in self.TRANSITIONS}
        for prev, curr in zip(rows, rows[1:]):
            p_state = str(prev["state"]).lower()
            c_state = str(curr["state"]).lower()
            for src, dst, key in self.TRANSITIONS:
                if p_state == src and c_state == dst:
                    counts[key] += 1

        funded_to_recovery = counts["funded_to_recovery"]
        recovery_to_funded = counts["recovery_to_funded"]
        recovery_rate = (
            round(recovery_to_funded / funded_to_recovery * 100.0, 1)
            if funded_to_recovery > 0
            else 100.0
        )
        return {**counts, "recovery_rate": recovery_rate}

    def recovery_events(self, *, limit: int = 10000) -> list[dict[str, Any]]:
        rows = self._rows(limit=limit)
        events: list[dict[str, Any]] = []
        event_no = 0
        idx = 0
        while idx < len(rows):
            row = rows[idx]
            if str(row["state"]).lower() != "recovery":
                idx += 1
                continue

            event_no += 1
            start_row = row
            start_idx = idx
            idx += 1
            while idx < len(rows) and str(rows[idx]["state"]).lower() == "recovery":
                idx += 1

            end_idx = min(idx, len(rows) - 1)
            end_row = rows[end_idx] if idx < len(rows) else rows[-1]
            recovered = idx < len(rows) and str(rows[idx]["state"]).lower() == "funded"
            end_ts = str(rows[idx]["timestamp"]) if idx < len(rows) else datetime.now(timezone.utc).isoformat()
            duration_days = round(_days_between(str(start_row["timestamp"]), end_ts), 1)

            events.append(
                {
                    "event_id": event_no,
                    "started": str(start_row["timestamp"])[:10],
                    "ended": end_ts[:10] if recovered else None,
                    "equity": start_row.get("equity"),
                    "dd_pct": start_row.get("drawdown_pct"),
                    "profile": start_row.get("profile"),
                    "duration_days": duration_days,
                    "recovered": recovered,
                }
            )
        return events

    def average_recovery_duration(self, *, limit: int = 10000) -> float:
        events = self.recovery_events(limit=limit)
        if not events:
            return 0.0
        total = sum(float(e["duration_days"]) for e in events)
        return round(total / len(events), 2)

    def funded_stability_score(self, *, limit: int = 10000) -> float:
        days = self.total_days_per_state(limit=limit)
        transitions = self.transition_counts(limit=limit)
        funded_days = float(days.get("funded", 0.0))
        if funded_days <= 0:
            return 0.0
        recovery_freq = transitions.get("funded_to_recovery", 0) / max(funded_days / 30.0, 1.0)
        score = max(0.0, min(100.0, 100.0 - recovery_freq * 20.0))
        return round(score, 1)

    def challenge_duration(self, *, limit: int = 10000) -> float:
        return self.total_days_per_state(limit=limit).get("challenge", 0.0)

    def profile_usage_statistics(self, *, limit: int = 10000) -> dict[str, int]:
        rows = self._rows(limit=limit)
        stats: dict[str, int] = {}
        for row in rows:
            profile = str(row.get("profile") or "UNKNOWN")
            stats[profile] = stats.get(profile, 0) + 1
        return dict(sorted(stats.items(), key=lambda kv: kv[1], reverse=True))

    def compute_health_score(self, *, limit: int = 10000) -> HealthScoreResult:
        transitions = self.transition_counts(limit=limit)
        days = self.total_days_per_state(limit=limit)
        rows = self._rows(limit=limit)

        recovery_events = transitions.get("funded_to_recovery", 0)
        recovery_penalty = min(30.0, recovery_events * 5.0)

        recent_dd = 0.0
        if rows:
            recent_dd = float(rows[-1].get("drawdown_pct") or 0.0)
        avg_dd = sum(float(r.get("drawdown_pct") or 0.0) for r in rows) / max(len(rows), 1)
        dd_penalty = min(25.0, max(recent_dd, avg_dd) * 2.5)

        funded_days = float(days.get("funded", 0.0))
        funded_bonus = min(15.0, funded_days / 30.0 * 3.0)

        challenge_success = 10.0 if transitions.get("challenge_to_funded", 0) > 0 else 0.0

        score = 100.0 - recovery_penalty - dd_penalty + funded_bonus + challenge_success
        score = max(0.0, min(100.0, score))

        if score >= 90.0:
            status = "EXCELLENT"
        elif score >= 75.0:
            status = "GOOD"
        elif score >= 60.0:
            status = "FAIR"
        else:
            status = "AT_RISK"

        return HealthScoreResult(
            score=round(score, 1),
            status=status,
            recovery_frequency_penalty=recovery_penalty,
            dd_penalty=dd_penalty,
            funded_duration_bonus=funded_bonus,
            challenge_success_bonus=challenge_success,
        )

    def build_summary(self, *, current_state: str, current_profile: str, limit: int = 10000) -> dict[str, Any]:
        days = self.total_days_per_state(limit=limit)
        transitions = self.transition_counts(limit=limit)
        health = self.compute_health_score(limit=limit)
        recovery_list = self.recovery_events(limit=limit)

        return {
            "current_state": str(current_state).upper(),
            "current_profile": current_profile,
            "health_score": health.score,
            "health_status": health.status,
            "health": health.to_dict(),
            "recovery_events": len(recovery_list),
            "recovery_event_reports": recovery_list,
            "average_recovery_duration_days": self.average_recovery_duration(limit=limit),
            "funded_days": round(days.get("funded", 0.0), 1),
            "challenge_days": round(days.get("challenge", 0.0), 1),
            "recovery_days": round(days.get("recovery", 0.0), 1),
            "live_days": round(days.get("live", 0.0), 1),
            "state_frequency": self.state_frequency(limit=limit),
            "transitions": transitions,
            "funded_stability_score": self.funded_stability_score(limit=limit),
            "profile_usage": self.profile_usage_statistics(limit=limit),
            "state_history_recent": self._repo.list_recent(limit=20),
        }
