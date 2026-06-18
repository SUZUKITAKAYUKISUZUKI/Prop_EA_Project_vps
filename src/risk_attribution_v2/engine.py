"""Portfolio Risk Attribution Engine v2 orchestrator."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.risk_attribution_v2.drawdown_attribution_engine import DrawdownAttributionEngine
from src.risk_attribution_v2.portfolio_health_engine import PortfolioHealthEngine
from src.risk_attribution_v2.recovery_cause_engine import RecoveryCauseEngine
from src.risk_attribution_v2.state_transition_engine import StateTransitionEngine
from src.risk_attribution_v2.strategy_risk_engine import StrategyRiskEngine
from src.risk_attribution_v2.symbol_risk_engine import SymbolRiskEngine
from src.risk_attribution_v2.weight_adjusted_contribution import WeightAdjustedContributionEngine
from src.state_analytics.state_analytics_engine import StateAnalyticsEngine
from src.state_analytics.state_history_repository import StateHistoryRepository


class PortfolioRiskAttributionEngineV2:
    """PRAE v2 — risk, recovery, and DD attribution across portfolio dimensions."""

    def __init__(
        self,
        *,
        history_repo: StateHistoryRepository | None = None,
        state_analytics: StateAnalyticsEngine | None = None,
    ) -> None:
        self._history_repo = history_repo or StateHistoryRepository(owns_connection=False)
        self._state_analytics = state_analytics or StateAnalyticsEngine(
            self._history_repo,
            owns_connection=False,
        )
        self._strategy_risk = StrategyRiskEngine()
        self._weight_contrib = WeightAdjustedContributionEngine()
        self._recovery = RecoveryCauseEngine(self._state_analytics)
        self._dd = DrawdownAttributionEngine()
        self._symbol = SymbolRiskEngine()
        self._transition = StateTransitionEngine(self._history_repo)
        self._health = PortfolioHealthEngine()

    def run_full_report(
        self,
        trades: pd.DataFrame,
        *,
        allocation_weights: dict[str, float] | None = None,
        state_health_score: float | None = None,
    ) -> dict[str, Any]:
        strategy_risk = self._strategy_risk.analyze(trades)
        weight_adjusted = self._weight_contrib.analyze(trades, allocation_weights=allocation_weights)
        recovery_analysis = self._recovery.analyze(trades)
        dd_attribution = self._dd.analyze(trades)
        symbol_risk = self._symbol.analyze(trades)
        state_transitions = self._transition.analyze(trades)

        if state_health_score is None:
            try:
                state_health_score = self._state_analytics.compute_health_score().score
            except Exception:
                state_health_score = None

        health_report = self._health.build_report(
            strategy_risk=strategy_risk,
            symbol_risk=symbol_risk,
            recovery_analysis=recovery_analysis,
            dd_attribution=dd_attribution,
            state_health_score=state_health_score,
        )

        top_risk_sources = [
            {"strategy": row["strategy"], "contribution_pct": row["risk_score"]}
            for row in strategy_risk[:5]
        ]

        return {
            "strategy_risk": strategy_risk,
            "weight_adjusted_contribution": weight_adjusted,
            "symbol_risk": symbol_risk,
            "recovery_analysis": recovery_analysis,
            "dd_attribution": dd_attribution,
            "state_transition_risk": state_transitions,
            "health_report": health_report,
            "top_risk_sources": top_risk_sources,
        }
