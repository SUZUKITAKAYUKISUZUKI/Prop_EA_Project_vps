"""PRAE v2 service layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prae.loaders import normalize_trade_frame
from src.api.dashboard_api import DEFAULT_PORTFOLIO_SOURCE
from src.repositories.base import normalize_source_path
from src.repositories.trade_repository import TradeRepository
from src.risk_attribution_v2.engine import PortfolioRiskAttributionEngineV2
from src.services.profile_service import ProfileService
from src.state_analytics.state_analytics_engine import StateAnalyticsEngine
from src.state_analytics.state_history_repository import StateHistoryRepository


class RiskAttributionV2Service:
    def __init__(self, *, owns_connections: bool = False) -> None:
        self._owns = owns_connections
        self._trades = TradeRepository(owns_connection=owns_connections)
        self._history = StateHistoryRepository(owns_connection=owns_connections)
        self._engine = PortfolioRiskAttributionEngineV2(
            history_repo=self._history,
            state_analytics=StateAnalyticsEngine(self._history, owns_connection=False),
        )

    def close(self) -> None:
        if self._owns:
            self._trades.close()
            self._history.close()

    def load_trades(
        self,
        *,
        source_path: str | Path | None = None,
        run_id: int | None = None,
    ) -> Any:
        import pandas as pd

        src = normalize_source_path(source_path or DEFAULT_PORTFOLIO_SOURCE)
        df = self._trades.get_trades(source_path=src, run_id=run_id, as_dataframe=True)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        return normalize_trade_frame(df, source=Path(src).name)

    def run_report(
        self,
        *,
        source_path: str | Path | None = None,
        run_id: int | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        trades = self.load_trades(source_path=source_path, run_id=run_id)
        allocations: dict[str, float] = {}
        pid = profile_id
        try:
            svc = ProfileService()
            ctx = svc.load_active_profile()
            allocations = ctx.strategy_allocations
            pid = pid or ctx.profile_id
            svc.close()
        except Exception:
            pass

        state_health: float | None = None
        try:
            state_health = StateAnalyticsEngine(self._history, owns_connection=False).compute_health_score().score
        except Exception:
            state_health = None

        report = self._engine.run_full_report(
            trades,
            allocation_weights=allocations,
            state_health_score=state_health,
        )
        report["source"] = normalize_source_path(source_path or DEFAULT_PORTFOLIO_SOURCE)
        report["profile_id"] = pid
        return report
