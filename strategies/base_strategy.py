"""
strategies/base_strategy.py — マルチ・レジーム戦略の共通抽象基底

v3.4: London Sweep Reversal (A) / London Continuation (B) 等が継承する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from strategies.base import StrategyResult


class BaseStrategy(ABC):
    """L1〜L3 セットアップ検出・特徴量算出の共通インターフェース。"""

    def __init__(self, weights_config: dict[str, int] | None = None, mode_h1: bool = False):
        self.weights = weights_config or {}
        self.mode_h1 = mode_h1

    @property
    @abstractmethod
    def setup_type(self) -> str:
        """L6 ログ / LLM コンテキスト用 setup_type 識別子。"""

    @abstractmethod
    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[Any]:
        """OHLCV からセットアップイベント一覧を検出。"""

    @abstractmethod
    def analyze_setup(
        self,
        setup: Any,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        """1 件のセットアップに対し candidate_score / raw_features を算出。"""

    def evaluate(self, market_data: dict[str, Any], account_state: dict[str, Any]) -> StrategyResult:
        """
        Live / 単体評価 API。

        market_data: OHLCV, spread_pips, bar_timestamp, active_setup 等
        account_state: daily_committed_risk_pct, profile 等（L0 AccountState スナップショット）
        """
        active = market_data.get("active_setup") or account_state.get("active_setup")
        if active is None:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
            )

        gbp_s = market_data.get("gbp_setup") or account_state.get("gbp_setup")
        eur_s = market_data.get("eur_setup") or account_state.get("eur_setup")
        h1_gbp = market_data.get("h1_gbp") or account_state.get("h1_gbp")
        h1_eur = market_data.get("h1_eur") or account_state.get("h1_eur")
        if h1_gbp is None or h1_eur is None:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
                raw_features={"reject_reason": "missing_h1_data"},
            )

        result = self.analyze_setup(active, gbp_s, eur_s, h1_gbp, h1_eur)
        return self._apply_account_guards(result, market_data, account_state)

    def _apply_account_guards(
        self,
        result: StrategyResult,
        market_data: dict[str, Any],
        account_state: dict[str, Any],
    ) -> StrategyResult:
        """サブクラスで L0 連携ガードを上書き可能。"""
        return result
