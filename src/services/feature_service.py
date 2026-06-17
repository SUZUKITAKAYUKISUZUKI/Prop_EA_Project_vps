"""Feature dataset service."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.repositories.feature_repository import FeatureRepository
from src.repositories.trade_repository import TradeRepository


class FeatureService:
    def __init__(
        self,
        feature_repo: FeatureRepository | None = None,
        trade_repo: TradeRepository | None = None,
    ) -> None:
        self.features = feature_repo or FeatureRepository()
        self.trades = trade_repo or TradeRepository(self.features._db)

    def load_dataset(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> pd.DataFrame:
        return self.features.get_bayes_dataset(run_id=run_id, source_path=str(source_path) if source_path else None)

    def load_bayes_dataset(
        self,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
        strategy: str | None = None,
    ) -> pd.DataFrame:
        return self.features.get_bayes_dataset(
            run_id=run_id,
            source_path=str(source_path) if source_path else None,
            strategy=strategy,
        )

    def load_strategy_dataset(
        self,
        strategy: str,
        *,
        run_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> pd.DataFrame:
        trades = self.trades.get_trades_by_strategy(strategy, run_id=run_id, source_path=source_path)
        features = self.load_bayes_dataset(run_id=run_id, source_path=source_path, strategy=strategy)
        if trades.empty:
            return features
        if features.empty:
            return trades
        return trades.merge(features, left_on="trade_id", right_on="trade_id", how="left", suffixes=("", "_feat"))
