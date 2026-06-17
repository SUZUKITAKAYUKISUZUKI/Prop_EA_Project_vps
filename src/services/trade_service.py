"""Trade statistics service."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.equity_engine import attach_equity_columns, build_equity_curve
from src.analytics.risk_metrics_engine import (
    expectancy,
    losing_streak,
    max_dd,
    profit_factor,
)
from src.repositories.cache_repository import CacheRepository
from src.repositories.trade_repository import TradeRepository


class TradeService:
    def __init__(
        self,
        trade_repo: TradeRepository | None = None,
        cache_repo: CacheRepository | None = None,
    ) -> None:
        self.trades = trade_repo or TradeRepository()
        self.cache = cache_repo or CacheRepository(self.trades._db)

    def _load(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
        with_equity: bool = False,
    ) -> pd.DataFrame:
        df = self.trades.get_trades_df(run_id=run_id, source_path=source_path)
        if with_equity:
            df = attach_equity_columns(df)
        return df

    def _cache_key(self, prefix: str, run_id: int | None, source_path: str | Path | None) -> str:
        if run_id is not None:
            return f"{prefix}:run:{run_id}"
        path = str(source_path or "latest")
        return f"{prefix}:src:{path}"

    def _stats(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> dict:
        return self.trades.get_trade_stats(run_id=run_id, source_path=source_path)

    def win_rate(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> float:
        key = self._cache_key("win_rate", run_id, source_path)

        def compute() -> float:
            return float(self._stats(run_id=run_id, source_path=source_path)["win_rate"])

        return float(self.cache.get_or_compute(key, compute))

    def pf(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> float:
        key = self._cache_key("pf", run_id, source_path)

        def compute() -> float:
            return float(self._stats(run_id=run_id, source_path=source_path)["pf"])

        return float(self.cache.get_or_compute(key, compute))

    def avg_r(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> float:
        return float(self._stats(run_id=run_id, source_path=source_path)["avg_r"])

    def max_dd(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> float:
        key = self._cache_key("max_dd", run_id, source_path)

        def compute() -> float:
            stats = self._stats(run_id=run_id, source_path=source_path)
            run = stats.get("run_id")
            if not run or stats["trades"] == 0:
                return 0.0
            if stats["trades"] > 20_000:
                row = self.trades._db.query(
                    """
                    SELECT entry_time, r_multiple FROM trades
                    WHERE run_id=? AND (result IS NULL OR result != 'NOT_EXECUTED')
                    ORDER BY entry_time ASC
                    """,
                    (int(run),),
                )
                df = pd.DataFrame([dict(r) for r in row])
                df["timestamp"] = pd.to_datetime(df["entry_time"])
                df["profit_r"] = df["r_multiple"]
                df["lot_factor"] = 1.0
                df["trade_result"] = "WIN"
                curve = build_equity_curve(df)
            else:
                df = self._load(run_id=run_id, source_path=source_path, with_equity=True)
                curve = build_equity_curve(df)
            if curve.empty:
                return 0.0
            return max_dd(curve["equity"])

        return float(self.cache.get_or_compute(key, compute))

    def losing_streak(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> int:
        stats = self._stats(run_id=run_id, source_path=source_path)
        run = stats.get("run_id")
        if not run:
            return 0
        row = self.trades._db.query(
            "SELECT result FROM trades WHERE run_id=? ORDER BY entry_time ASC",
            (int(run),),
        )
        return losing_streak(pd.Series([r["result"] for r in row]))

    def summary(self, *, run_id: int | None = None, source_path: str | Path | None = None) -> dict:
        key = self._cache_key("trade_summary", run_id, source_path)

        def compute() -> dict:
            stats = self._stats(run_id=run_id, source_path=source_path)
            return {
                "trades": int(stats["trades"]),
                "win_rate": float(stats["win_rate"]),
                "pf": float(stats["pf"]),
                "avg_r": float(stats["avg_r"]),
                "total_r": float(stats["total_r"]),
                "max_dd": self.max_dd(run_id=run_id, source_path=source_path),
                "losing_streak": self.losing_streak(run_id=run_id, source_path=source_path),
            }

        return dict(self.cache.get_or_compute(key, compute))
