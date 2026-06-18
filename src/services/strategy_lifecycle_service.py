"""Strategy Lifecycle Manager service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.api.risk_attribution_v2_api import get_risk_attribution_v2
from src.api.state_analytics_api import get_account_state_summary
from src.services.profile_service import ProfileService
from src.services.risk_attribution_v2_service import RiskAttributionV2Service
from src.strategy_lifecycle.engine import StrategyLifecycleEngine
from src.strategy_lifecycle.lifecycle_report import LifecycleReport


class StrategyLifecycleService:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._engine = StrategyLifecycleEngine(owns_connections=owns_connections)
        self._profiles = ProfileService()
        self._trades = RiskAttributionV2Service(owns_connections=False)
        self._reporter = LifecycleReport()

    def close(self) -> None:
        if self._owns:
            self._engine.close()
        self._profiles.close()
        self._trades.close()

    def _context(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any], Any, str]:
        ctx = self._profiles.load_active_profile()
        pid = profile_id or ctx.profile_id
        prae_v2 = get_risk_attribution_v2(source_path=source_path, profile_id=pid)
        state_summary = get_account_state_summary(
            current_state=str(ctx.settings.get("account_state") or ""),
            current_profile=pid,
        )
        trades = self._trades.load_trades(source_path=source_path)
        return ctx, prae_v2, state_summary, trades, pid

    def get_strategy_lifecycle(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        ctx, prae_v2, state_summary, trades, pid = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.get_lifecycle_dashboard(
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
            profile_id=pid,
        )

    def get_portfolio_fit_score(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, _ = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.get_portfolio_fit_score(
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
        )

    def get_strategy_portfolio_fit(
        self,
        strategy_id: str,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, _ = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.get_strategy_portfolio_fit(
            strategy_id,
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
        )

    def get_portfolio_fit_ranking(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, _ = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.get_portfolio_fit_ranking(
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
        )

    def evaluate_strategy(
        self,
        strategy_id: str,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, pid = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        result = self._engine.evaluate_strategy(
            strategy_id,
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
            profile_id=pid,
        )
        return self._reporter.evaluate_response(result)

    def promote_strategy(
        self,
        strategy_id: str,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, pid = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        result = self._engine.promote_strategy(
            strategy_id,
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
            profile_id=pid,
            force=force,
        )
        return self._reporter.evaluate_response(result) | {"promoted": result.get("promoted", False)}

    def retire_strategy(self, strategy_id: str, *, reason: str = "manual_retire") -> dict[str, Any]:
        return self._engine.retire_strategy(strategy_id, reason=reason)

    def run_weekly_evaluation(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, pid = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.run_weekly_evaluation(
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
            profile_id=pid,
            force=force,
        )

    def apply_lifecycle_weights(self, weights: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
        return self._engine.apply_lifecycle_to_weights(weights)

    def get_strategy_genealogy(self, strategy_id: str | None = None) -> dict[str, Any]:
        return self._engine.get_strategy_genealogy(strategy_id)

    def get_strategy_explanation(
        self,
        strategy_id: str,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, pid = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.get_strategy_explanation(
            strategy_id,
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
            profile_id=pid,
        )

    def get_core_strategies(self) -> list[dict[str, Any]]:
        return self._engine.get_core_strategies()

    def get_lifecycle_decision_report(
        self,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> list[dict[str, Any]]:
        _, prae_v2, state_summary, trades, pid = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.get_lifecycle_decision_report(
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
            profile_id=pid,
        )

    def get_strategy_governance_context(
        self,
        strategy_id: str,
        *,
        source_path: str | Path | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        _, prae_v2, state_summary, trades, pid = self._context(
            source_path=source_path,
            profile_id=profile_id,
        )
        return self._engine.get_strategy_governance_context(
            strategy_id,
            prae_v2=prae_v2,
            state_summary=state_summary,
            trades=trades,
            profile_id=pid,
        )
