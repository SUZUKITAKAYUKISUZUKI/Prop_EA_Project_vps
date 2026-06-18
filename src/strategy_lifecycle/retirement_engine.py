"""Strategy retirement and reinstatement rules for SLM."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class RetirementEngine:
    RECOVERY_MAX_DAYS = 90.0

    def recovery_to_retired(self, metrics: dict[str, Any], *, recovery_started_at: str | None = None) -> tuple[bool, str]:
        if metrics.get("pf", 1.0) < 1.0:
            return True, "pf_below_1"
        if metrics.get("score", 100.0) < 40.0:
            return True, "score_below_40"
        if metrics.get("oos_pf", 1.0) < 1.0:
            return True, "oos_degraded"
        if metrics.get("portfolio_fit_score", 100.0) < 20.0:
            return True, "portfolio_fit_below_20"
        if recovery_started_at:
            try:
                started = datetime.fromisoformat(recovery_started_at.replace("Z", "+00:00"))
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                days = (datetime.now(timezone.utc) - started).total_seconds() / 86400.0
                if days > self.RECOVERY_MAX_DAYS and metrics.get("score", 0.0) < 60.0:
                    return True, "recovery_failed_90d"
            except ValueError:
                pass
        return False, "recovery_ongoing"

    def retired_to_incubation(self, *, strategy_version: str, previous_version: str) -> tuple[bool, str]:
        if strategy_version != previous_version:
            return True, "strategy_version_changed"
        return False, "retired_unchanged"

    def next_stage(
        self,
        current: str,
        metrics: dict[str, Any],
        *,
        recovery_started_at: str | None = None,
        strategy_version: str = "1.0",
        previous_version: str = "1.0",
    ) -> tuple[str | None, str, bool]:
        if current == "RECOVERY":
            ok, reason = self.recovery_to_retired(metrics, recovery_started_at=recovery_started_at)
            return ("RETIRED", reason, ok) if ok else (None, reason, False)
        if current == "RETIRED":
            ok, reason = self.retired_to_incubation(
                strategy_version=strategy_version,
                previous_version=previous_version,
            )
            return ("INCUBATION", reason, ok) if ok else (None, reason, False)
        return (None, "no_retirement_path", False)
