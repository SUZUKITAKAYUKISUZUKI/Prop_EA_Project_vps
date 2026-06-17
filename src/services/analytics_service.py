"""BT / WFT / MC analytics service."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.equity_engine import build_equity_curve
from src.analytics.risk_metrics_engine import profit_factor, recovery_factor, sharpe_ratio
from src.repositories.cache_repository import CacheRepository
from src.repositories.portfolio_repository import PortfolioRepository
from src.repositories.trade_repository import TradeRepository
from src.services.trade_service import TradeService


class AnalyticsService:
    def __init__(
        self,
        trade_service: TradeService | None = None,
        portfolio_repo: PortfolioRepository | None = None,
        cache_repo: CacheRepository | None = None,
    ) -> None:
        self.trade_service = trade_service or TradeService()
        self.portfolio = portfolio_repo or PortfolioRepository(self.trade_service.trades._db)
        self.cache = cache_repo or CacheRepository(self.trade_service.trades._db)

    def calculate_pf(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> float:
        return self.trade_service.pf(run_id=run_id, source_path=source_path)

    def calculate_sharpe(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> float:
        key = f"sharpe:{run_id}:{source_path}"

        def compute() -> float:
            df = self.trade_service._load(run_id=run_id, source_path=source_path)
            return sharpe_ratio(df["profit_r"])

        return float(self.cache.get_or_compute(key, compute))

    def calculate_recovery(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> float:
        summary = self.trade_service.summary(run_id=run_id, source_path=source_path)
        return recovery_factor(summary["total_r"], summary["max_dd"])

    def calculate_stability(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> float:
        wft = self.portfolio.get_wft_results(run_id=run_id, source_path=str(source_path) if source_path else None)
        if not wft:
            return 0.0
        flags = [row.get("pass_flag") for row in wft if row.get("pass_flag") is not None]
        if not flags:
            return 0.0
        return float(sum(flags) / len(flags) * 100.0)

    def bt_summary(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> dict:
        key = f"bt_summary:{run_id}:{source_path}"

        def compute() -> dict:
            stored = self.portfolio.get_bt_summary(run_id=run_id, source_path=str(source_path) if source_path else None)
            live = self.trade_service.summary(run_id=run_id, source_path=source_path)
            if stored:
                return {
                    "pf": stored.get("pf") or live["pf"],
                    "wr": stored.get("wr") or live["win_rate"],
                    "total_r": stored.get("total_r") or live["total_r"],
                    "max_dd": stored.get("max_dd") or live["max_dd"],
                    "sharpe": stored.get("sharpe") or self.calculate_sharpe(run_id=run_id, source_path=source_path),
                    "recovery": stored.get("recovery") or self.calculate_recovery(run_id=run_id, source_path=source_path),
                }
            return {
                "pf": live["pf"],
                "wr": live["win_rate"],
                "total_r": live["total_r"],
                "max_dd": live["max_dd"],
                "sharpe": self.calculate_sharpe(run_id=run_id, source_path=source_path),
                "recovery": self.calculate_recovery(run_id=run_id, source_path=source_path),
            }

        return dict(self.cache.get_or_compute(key, compute))
