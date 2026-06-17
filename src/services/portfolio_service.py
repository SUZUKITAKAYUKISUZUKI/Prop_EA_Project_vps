"""Portfolio analytics service."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.correlation_engine import strategy_correlation
from src.analytics.risk_metrics_engine import profit_factor
from src.repositories.cache_repository import CacheRepository
from src.repositories.portfolio_repository import PortfolioRepository
from src.repositories.trade_repository import TradeRepository


class PortfolioService:
    def __init__(
        self,
        trade_repo: TradeRepository | None = None,
        portfolio_repo: PortfolioRepository | None = None,
        cache_repo: CacheRepository | None = None,
    ) -> None:
        self.trades = trade_repo or TradeRepository()
        self.portfolio = portfolio_repo or PortfolioRepository(self.trades._db)
        self.cache = cache_repo or CacheRepository(self.trades._db)

    def allocation(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> dict[str, float]:
        return self.portfolio.get_latest_allocation(
            run_id=run_id,
            source_path=str(source_path) if source_path else None,
        )

    def correlation_matrix(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> pd.DataFrame:
        key = f"corr:{run_id}:{source_path}"

        def compute() -> dict:
            df = self.trades.get_trades_df(run_id=run_id, source_path=source_path)
            corr = strategy_correlation(df)
            return corr.to_dict()

        cached = self.cache.get_or_compute(key, compute)
        return pd.DataFrame(cached)

    def risk_contribution(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> list[dict]:
        rows = self.portfolio.get_risk_attribution(
            run_id=run_id,
            source_path=str(source_path) if source_path else None,
        )
        if rows:
            return rows
        try:
            df = self.trades.get_trades_df(run_id=run_id, source_path=source_path)
        except FileNotFoundError:
            return []
        if df.empty:
            return []
        out: list[dict] = []
        for strategy, sub in df.groupby("setup_type"):
            r_vals = sub["profit_r"]
            out.append(
                {
                    "strategy": strategy,
                    "contribution_r": float(r_vals.sum()),
                    "contribution_dd": None,
                    "contribution_pf": profit_factor(r_vals),
                }
            )
        return sorted(out, key=lambda x: x["contribution_r"], reverse=True)

    def strategy_ranking(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> list[dict]:
        stats = self.trades.get_strategy_stats(run_id=run_id, source_path=source_path)
        if stats:
            return stats
        df = self.trades.get_trades_df(run_id=run_id, source_path=source_path)
        rows: list[dict] = []
        for strategy, sub in df.groupby("setup_type"):
            r_vals = sub["profit_r"]
            rows.append(
                {
                    "strategy": strategy,
                    "trades": len(sub),
                    "total_r": float(r_vals.sum()),
                    "pf": profit_factor(r_vals),
                    "avg_r": float(r_vals.mean()),
                }
            )
        return sorted(rows, key=lambda x: x["total_r"], reverse=True)

    def summary(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> dict:
        key = f"portfolio_summary:{run_id}:{source_path}"

        def compute() -> dict:
            allocation = self.allocation(run_id=run_id, source_path=source_path)
            risk = self.risk_contribution(run_id=run_id, source_path=source_path)
            try:
                ranking = self.strategy_ranking(run_id=run_id, source_path=source_path)
            except FileNotFoundError:
                ranking = []
            top = self.portfolio.get_portfolio_result(run_id=run_id, source_path=str(source_path) if source_path else None)
            return {
                "allocation": allocation,
                "strategy_ranking": ranking,
                "risk_contribution": risk,
                "portfolio_result": top,
            }

        return dict(self.cache.get_or_compute(key, compute))
