"""Recovery cause analysis using account_state_history + trade ledger."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.risk_attribution_v2.common import loss_contribution, normalize_contribution_pct, prepare_trades
from src.state_analytics.state_analytics_engine import StateAnalyticsEngine


class RecoveryCauseEngine:
    PRE_TRADE_WINDOW = 20

    def __init__(self, state_analytics: StateAnalyticsEngine | None = None) -> None:
        self._state = state_analytics or StateAnalyticsEngine(owns_connection=False)

    def analyze(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        work = prepare_trades(trades)
        events = self._state.recovery_events()
        if work.empty or not events:
            return []

        reports: list[dict[str, Any]] = []
        for event in events:
            start_ts = pd.to_datetime(event["started"], utc=True)
            prior = work[work["timestamp"] < start_ts].tail(self.PRE_TRADE_WINDOW)
            if prior.empty:
                continue

            strategy_losses = prior.loc[prior["R"] < 0].groupby("strategy_code")["R"].sum()
            loss_map = {str(k): abs(float(v)) for k, v in strategy_losses.items()}
            strategy_contrib = normalize_contribution_pct(loss_map)

            symbol_contrib = loss_contribution(prior, "pair")
            direction_contrib = loss_contribution(prior, "direction")
            session_contrib = loss_contribution(prior, "session")

            top_strategy = max(strategy_contrib, key=strategy_contrib.get) if strategy_contrib else None
            reports.append(
                {
                    "event_id": event["event_id"],
                    "started": event["started"],
                    "dd_pct": event.get("dd_pct"),
                    "profile": event.get("profile"),
                    "duration_days": event.get("duration_days"),
                    "recovered": event.get("recovered"),
                    "top_cause": top_strategy,
                    "strategy_contribution": strategy_contrib,
                    "symbol_contribution": symbol_contrib,
                    "direction_contribution": direction_contrib,
                    "session_contribution": session_contrib,
                    "recovery_attribution_pct": strategy_contrib,
                    "recovery_attribution_r": {
                        str(k): round(float(v), 2)
                        for k, v in strategy_losses.items()
                    },
                }
            )
        return reports
