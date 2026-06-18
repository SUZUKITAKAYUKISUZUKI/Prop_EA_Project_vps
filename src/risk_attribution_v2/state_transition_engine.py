"""State transition risk attribution for PRAE v2."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.risk_attribution_v2.common import loss_contribution, prepare_trades
from src.state_analytics.state_history_repository import StateHistoryRepository


class StateTransitionEngine:
    PRE_TRADE_WINDOW = 20
    TRACKED_TRANSITIONS = (
        ("challenge", "funded"),
        ("funded", "recovery"),
        ("recovery", "funded"),
    )

    def __init__(self, repo: StateHistoryRepository | None = None) -> None:
        self._repo = repo or StateHistoryRepository(owns_connection=False)

    def analyze(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        work = prepare_trades(trades)
        history = self._repo.list_history(limit=10000)
        if work.empty or len(history) < 2:
            return []

        results: list[dict[str, Any]] = []
        for prev, curr in zip(history, history[1:]):
            from_state = str(prev.get("state", "")).lower()
            to_state = str(curr.get("state", "")).lower()
            if (from_state, to_state) not in self.TRACKED_TRANSITIONS:
                continue

            ts = pd.to_datetime(curr["timestamp"])
            prior = work[work["timestamp"] < ts].tail(self.PRE_TRADE_WINDOW)
            if prior.empty:
                continue

            strategy_contrib = loss_contribution(prior, "strategy_code")
            top = max(strategy_contrib, key=strategy_contrib.get) if strategy_contrib else None
            results.append(
                {
                    "transition": f"{from_state}_to_{to_state}",
                    "timestamp": str(curr["timestamp"]),
                    "from_state": from_state,
                    "to_state": to_state,
                    "profile": curr.get("profile"),
                    "drawdown_pct": curr.get("drawdown_pct"),
                    "top_strategy": top,
                    "strategy_contribution": strategy_contrib,
                }
            )
        return results
