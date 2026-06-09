"""
strategies/london_sweep_failure.py — London Sweep Failure Continuation (Strategy A / LSFC)

本番メイン軸。Sweep 後の反転失敗を検知し、ブレイクアウト方向への継続（順張り）に便乗する。
ロンドンセッション外では検知・執行を行わない（ルックアヘッドなし）。

BT / 本番の正式足構成: 構造足 H1 / 執行足 M15（`backtest_runner --bar-minutes 15`）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import analyze_htf_trend, is_counter_trend
from strategies.market_utils import (
    PIP_SIZE,
    LONDON_SESSION_HOUR_END,
    LONDON_SESSION_HOUR_START,
    SMTFeatures,
    calc_smt_features,
    compute_atr,
    positional_index as _positional_index,
)
from strategies.session_dst import DATA_DST_TYPE, shift_hour_range

SETUP_TYPE = "LONDON_SWEEP_FAILURE_CONTINUATION"
MIN_RR = 2.0


@dataclass(frozen=True)
class LsfcConfig:
    lookback_candles: int = 20
    failure_window: int = 4
    retracement_atr_ratio: float = 0.5
    trigger_offset_pips: float = 1.0


def load_lsfc_config() -> LsfcConfig:
    """Optuna / 環境変数から LSFC パラメータを読み込む。"""
    return LsfcConfig(
        lookback_candles=int(os.getenv("LSFC_LOOKBACK_CANDLES", "20")),
        failure_window=int(os.getenv("LSFC_FAILURE_WINDOW", "4")),
        retracement_atr_ratio=float(os.getenv("LSFC_RETRACEMENT_ATR_RATIO", "0.5")),
        trigger_offset_pips=float(os.getenv("LSFC_TRIGGER_OFFSET_PIPS", "1.0")),
    )


def is_lsfc_l4_bypass() -> bool:
    """LSFC L4 Gemini バイパス（1=RULE_BASE_ONLY / 0=Gemini L4 有効）。"""
    return os.getenv("LSFC_L4_BYPASS", "1").strip().lower() in ("1", "true", "yes", "on")


def _resolve_lsfc_session_hours(
    session_date: date,
    dst_type: str = DATA_DST_TYPE,
) -> range:
    """ロンドン監視帯（Sweep Reversal と同一: 15–20、DST 時 -1h）。"""
    return shift_hour_range(
        session_date,
        LONDON_SESSION_HOUR_START,
        LONDON_SESSION_HOUR_END,
        dst_type,
    )


@dataclass
class LsfcSetup:
    """LSFC 1 件のセットアップ。"""

    timestamp: pd.Timestamp
    pair: str
    direction: str
    pool_high: float
    pool_low: float
    sweep_level: float
    sweep_extreme: float
    sweep_bar_index: int
    failure_extreme: float
    failure_retracement_depth: float
    sweep_high_low_distance_atr: float
    entry_price: float
    stop_loss: float
    take_profit: float
    sweep_distance: float
    atr: float
    bar_index: int


def _atr_at_bar(
    atr_series: pd.Series,
    structure_df: pd.DataFrame,
    bar: pd.Series,
    bar_index: int,
) -> float:
    if bar_index < len(atr_series) and pd.notna(atr_series.iloc[bar_index]):
        return float(atr_series.iloc[bar_index])
    match_idx = structure_df.index[structure_df["datetime"] == bar["datetime"]]
    if len(match_idx) > 0:
        pos = _positional_index(structure_df, match_idx[0])
        if pos < len(atr_series) and pd.notna(atr_series.iloc[pos]):
            return float(atr_series.iloc[pos])
    return max(float(bar["high"] - bar["low"]), PIP_SIZE * 10)


def _pool_levels(df: pd.DataFrame, bar_index: int, lookback: int) -> tuple[float, float] | None:
    """直前 lookback 本（現バー除外）のスイング高安。"""
    if bar_index < lookback:
        return None
    window = df.iloc[bar_index - lookback : bar_index]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min())


def _bar_in_session(bar: pd.Series, session_date: date, london_hours: range) -> bool:
    ts = pd.Timestamp(bar["datetime"])
    return ts.date() == session_date and int(bar["hour"]) in london_hours


def _detect_failure_pattern(
    df: pd.DataFrame,
    sweep_index: int,
    direction: str,
    sweep_level: float,
    failure_window: int,
    max_depth_price: float,
    session_date: date,
    london_hours: range,
) -> tuple[bool, float, float, int]:
    """
    Phase B: 反転失敗パターン。

    Returns:
        (confirmed, failure_retracement_depth, failure_extreme, turn_back_index)
    """
    window_end = min(sweep_index + failure_window, len(df) - 1)
    retracement_seen = False
    failure_depth = 0.0
    failure_extreme: float | None = None
    deepest_bar = sweep_index
    turn_back_index = -1

    for j in range(sweep_index + 1, window_end + 1):
        bar_j = df.iloc[j]
        if not _bar_in_session(bar_j, session_date, london_hours):
            break

        if direction == "BUY":
            if float(bar_j["low"]) < sweep_level:
                retracement_seen = True
                pull = float(bar_j["low"])
                depth = sweep_level - pull
                if depth > failure_depth:
                    failure_depth = depth
                    deepest_bar = j
                failure_extreme = pull if failure_extreme is None else min(failure_extreme, pull)
        else:
            if float(bar_j["high"]) > sweep_level:
                retracement_seen = True
                pull = float(bar_j["high"])
                depth = pull - sweep_level
                if depth > failure_depth:
                    failure_depth = depth
                    deepest_bar = j
                failure_extreme = pull if failure_extreme is None else max(failure_extreme, pull)

    if not retracement_seen or failure_depth <= 0 or failure_depth > max_depth_price:
        return False, 0.0, 0.0, -1
    if failure_extreme is None:
        return False, 0.0, 0.0, -1

    for j in range(deepest_bar, window_end + 1):
        bar_j = df.iloc[j]
        if not _bar_in_session(bar_j, session_date, london_hours):
            break
        if direction == "BUY" and float(bar_j["close"]) > sweep_level:
            turn_back_index = j
            break
        if direction == "SELL" and float(bar_j["close"]) < sweep_level:
            turn_back_index = j
            break

    if turn_back_index < 0:
        return False, 0.0, 0.0, -1

    return True, failure_depth, failure_extreme, turn_back_index


def _scan_trigger_bar(
    df: pd.DataFrame,
    start_index: int,
    direction: str,
    trigger_level: float,
    session_date: date,
    london_hours: range,
) -> int | None:
    """Phase C: 継続トリガーバーを探索（ロンドン帯内のみ）。"""
    for j in range(start_index, len(df)):
        bar_j = df.iloc[j]
        if pd.Timestamp(bar_j["datetime"]).date() != session_date:
            break
        if int(bar_j["hour"]) not in london_hours:
            continue

        if direction == "BUY":
            if float(bar_j["close"]) >= trigger_level or float(bar_j["high"]) >= trigger_level:
                return j
        elif float(bar_j["close"]) <= trigger_level or float(bar_j["low"]) <= trigger_level:
            return j
    return None


def _build_lsfc_setup(
    df: pd.DataFrame,
    trigger_index: int,
    pair_name: str,
    direction: str,
    pool_high: float,
    pool_low: float,
    sweep_level: float,
    sweep_extreme: float,
    sweep_bar_index: int,
    failure_extreme: float,
    failure_depth: float,
    sweep_hl_distance_atr: float,
    atr_val: float,
    structure_df: pd.DataFrame,
    trigger_offset_pips: float,
) -> LsfcSetup | None:
    bar = df.iloc[trigger_index]
    entry = float(bar["close"])
    offset = trigger_offset_pips * PIP_SIZE

    if direction == "BUY":
        stop_loss = min(failure_extreme, float(bar["open"]))
        risk = entry - stop_loss
        if risk <= 0:
            return None
        take_profit = entry + MIN_RR * risk
        sweep_distance = sweep_extreme - sweep_level
    else:
        stop_loss = max(failure_extreme, float(bar["open"]))
        risk = stop_loss - entry
        if risk <= 0:
            return None
        take_profit = entry - MIN_RR * risk
        sweep_distance = sweep_level - sweep_extreme

    match_idx = structure_df.index[structure_df["datetime"] == bar["datetime"]]
    if len(match_idx) > 0:
        bar_index = _positional_index(structure_df, match_idx[0])
    else:
        bar_index = _positional_index(structure_df, trigger_index)
    bar_index = min(max(bar_index, 0), len(structure_df) - 1)

    return LsfcSetup(
        timestamp=pd.Timestamp(bar["datetime"]),
        pair=pair_name,
        direction=direction,
        pool_high=pool_high,
        pool_low=pool_low,
        sweep_level=sweep_level,
        sweep_extreme=sweep_extreme,
        sweep_bar_index=sweep_bar_index,
        failure_extreme=failure_extreme,
        failure_retracement_depth=failure_depth,
        sweep_high_low_distance_atr=sweep_hl_distance_atr,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        sweep_distance=sweep_distance,
        atr=atr_val,
        bar_index=bar_index,
    )


def detect_london_sweep_failure_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    config: LsfcConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> list[LsfcSetup]:
    """
    日次・ロンドン帯内で LSFC パターン（Sweep → Failure → Continuation）を検出。
    各日最大 1 セットアップ（先着）。numpy hot path。
    """
    from strategies.bt_ohlcv import as_ohlcv
    from strategies.lsfc_scan_hot import detect_london_sweep_failure_setups_np

    if len(df) == 0:
        return []
    exec_arr = as_ohlcv(df)
    struct_arr = as_ohlcv(h1_df if h1_df is not None else df)
    return detect_london_sweep_failure_setups_np(
        exec_arr,
        struct_arr,
        pair_name,
        config=config,
        progress_hook=progress_hook,
    )



def calc_lsfc_candidate_score(
    setup: LsfcSetup,
    gbp_setup: LsfcSetup | None,
    eur_setup: LsfcSetup | None,
    htf_aligned: bool,
) -> float:
    """L2 足切り用 0–100 スコア。"""
    score = 0.0

    if gbp_setup and eur_setup:
        score += 20.0 if gbp_setup.direction == eur_setup.direction else 8.0
    elif gbp_setup or eur_setup:
        score += 5.0

    sweep_atr = setup.sweep_distance / setup.atr if setup.atr > 0 else 0.0
    score += min(25.0, sweep_atr * 20.0)

    retrace_atr = (
        setup.failure_retracement_depth / setup.atr if setup.atr > 0 else 0.0
    )
    shallow_quality = max(0.0, 1.0 - retrace_atr)
    score += shallow_quality * 20.0

    score += min(15.0, setup.sweep_high_low_distance_atr * 5.0)

    if htf_aligned:
        score += 15.0

    return round(max(0.0, min(100.0, score)), 2)


class LondonSweepFailureStrategy(BaseStrategy):
    """Strategy A (LSFC): Sweep 反転失敗 → 継続順張り — 本番メイン軸。"""

    def __init__(
        self,
        weights_config: dict[str, int] | None = None,
        mode_h1: bool = False,
        lsfc_config: LsfcConfig | None = None,
    ):
        super().__init__(weights_config, mode_h1)
        self.lsfc_config = lsfc_config or load_lsfc_config()
        self._htf_gbp: pd.DataFrame | None = None
        self._htf_eur: pd.DataFrame | None = None

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[LsfcSetup]:
        return detect_london_sweep_failure_setups(
            df, pair_name, h1_df, self.lsfc_config
        )

    def analyze_setup(
        self,
        setup: LsfcSetup,
        gbp_setup: LsfcSetup | None,
        eur_setup: LsfcSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        h1_ref = h1_gbp if setup.pair == "GBPUSD" else h1_eur
        htf_df = self._htf_gbp if setup.pair == "GBPUSD" else self._htf_eur
        htf_result = analyze_htf_trend(
            h1_ref,
            setup.timestamp,
            htf_df=htf_df,
        )
        htf_trend_direction = htf_result.direction
        counter = is_counter_trend(setup.direction, htf_result.direction)
        htf_aligned = not counter and htf_result.direction != "NEUTRAL"

        smt_feats = calc_smt_features(gbp_setup, eur_setup)
        candidate_score = calc_lsfc_candidate_score(
            setup, gbp_setup, eur_setup, htf_aligned
        )
        atr_ratio = setup.sweep_distance / setup.atr if setup.atr > 0 else 0.0
        failure_depth_atr = (
            setup.failure_retracement_depth / setup.atr if setup.atr > 0 else 0.0
        )

        raw_features: dict[str, Any] = {
            "smt_intensity": smt_feats.intensity,
            "smt_diff": smt_feats.diff,
            "smt_leader": smt_feats.leader,
            "wick_ratio_pct": 0.0,
            "atr_ratio": round(atr_ratio, 4),
            "has_bos": False,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf_trend_direction,
            "sweep_high_low_distance_atr": round(setup.sweep_high_low_distance_atr, 4),
            "failure_retracement_depth": round(failure_depth_atr, 4),
            "pool_high": setup.pool_high,
            "pool_low": setup.pool_low,
            "sweep_level": setup.sweep_level,
            "failure_extreme": setup.failure_extreme,
            "htf_bypass": True,
            "htf_would_block": counter,
            "l4_bypass": True,
        }

        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=candidate_score,
            raw_features=raw_features,
        )

    def evaluate(self, payload: dict, state: dict) -> StrategyResult:
        active: LsfcSetup | None = state.get("active_setup")
        if active is None:
            return StrategyResult(is_setup=False, setup_type=self.setup_type, direction="")

        gbp_s = state.get("gbp_setup")
        eur_s = state.get("eur_setup")
        return self.analyze_setup(
            active,
            gbp_s,
            eur_s,
            state["h1_gbp"],
            state["h1_eur"],
        )
