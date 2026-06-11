"""
strategies/ttm.py — TTMS (TTM Short, Flow-based EV)

LOW_UPDATE × SHORT 特化。フロー系特徴量 → Naive Bayes EV → 可変サイジング。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.bt_ohlcv import BtOhlcvFrame, as_ohlcv, resample_bars_ns
from strategies.market_utils import JPY_PIP_SIZE, pip_size_for_pair
from strategies.ttm_arrays import (
    EVENT_END_MIN,
    EVENT_START_MIN,
    filter_ttm_event_window,
)
from strategies.ttm_scan_engine import scan_ttm_setups_from_frames

logger = logging.getLogger(__name__)

SETUP_TYPE = "TTM_LIQUIDITY_EVENT"
STRATEGY_ABBREV = "TTMS"
STRATEGY_FULL_NAME = "TTM Short"
TTM_PAIR_PRIMARY = "USDJPY"
ALLOWED_PAIRS = frozenset({TTM_PAIR_PRIMARY})
TTM_SHORT_EVENT_TYPE = "LOW_UPDATE"
TTM_SHORT_TRIGGER = "ASIA_LOW_UPDATE"

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_INPUT_TZ = ZoneInfo(os.getenv("TTM_INPUT_TZ", "UTC"))

EVENT_START_JST = time(8, 30)
EVENT_END_JST = time(10, 30)
TTM_TIME_JST = time(9, 55)

MAX_EVENTS_PER_DAY = int(os.getenv("TTM_MAX_EVENTS_PER_DAY", "8"))
BAR_NS_5M = int(np.timedelta64(5, "m") / np.timedelta64(1, "ns"))
DEFAULT_SCAN_BAR_MINUTES = 1

TTM_BT_WINDOW_START_NS = int(np.datetime64("2021-01-01T00:00:00", "ns").astype(np.int64))
TTM_BT_WINDOW_END_NS = int(np.datetime64("2025-12-31T23:59:59", "ns").astype(np.int64))


def clip_ttm_bt_window(frame: BtOhlcvFrame) -> BtOhlcvFrame:
    return frame.clip_window(TTM_BT_WINDOW_START_NS, TTM_BT_WINDOW_END_NS)


Direction = Literal["SHORT"]
PatternClass = Literal["TTM_LOW_UPDATE"]
EventTrigger = Literal["ASIA_LOW_UPDATE"]
EventType = Literal["LOW_UPDATE"]

TTM_EVENT_TYPES: tuple[str, ...] = (TTM_SHORT_EVENT_TYPE,)
TTM_TRUST_SCORE = 100


def derive_ttm_event_type(*, event_trigger: str, **_kwargs: Any) -> str:
    del _kwargs
    return TTM_SHORT_EVENT_TYPE


def _minute_bucket(hour: int, minute: int) -> str:
    if hour < 8 or (hour == 8 and minute < 30):
        return "pre_event"
    if hour == 8 and minute < 45:
        return "08:30-08:44"
    if hour == 8 or (hour == 9 and minute < 30):
        return "08:45-09:29"
    if hour == 9 and minute < 55:
        return "09:30-09:54"
    if hour == 9 or (hour == 10 and minute <= 30):
        return "09:55-10:30"
    return "post_event"


def _session_label(hour: int) -> str:
    if 0 <= hour < 8:
        return "ASIA"
    if 8 <= hour < 10:
        return "TTM_EVENT"
    return "POST_TTM"


def _week_of_month(day: int) -> int:
    return (day - 1) // 7 + 1


def is_ttm_pure_data_mode() -> bool:
    """TTMS: L0-L6 防御オフ（常時）。BT / live 共通。"""
    return True


def is_ttm_defense_pure_mode() -> bool:
    """TTMS: 防御レイヤー純粋モード（常時）。"""
    return True


def is_ttm_l4_bypass() -> bool:
    """TTMS: Gemini L4 監査は常に無効。"""
    return True


@dataclass(frozen=True)
class TtmFeatures:
    pair: str
    timestamp: str
    weekday: int
    month: int
    quarter: int
    week_of_month: int
    hour: int
    minute: int
    minute_bucket: str
    session: str
    minutes_to_ttm: float
    minutes_after_ttm: float
    is_gotobi: bool
    is_month_end: bool
    is_quarter_end: bool
    pre_ttm_return: float
    pre_ttm_velocity: float
    pre_ttm_range: float
    pre_ttm_atr_ratio: float
    asian_range: float
    asian_low_distance: float
    asian_range_pct: float
    low_break_distance: float
    low_break_velocity: float
    atr_m5: float
    atr_m15: float
    atr_h1: float
    atr_ratio_m5_h1: float
    atr_ratio_m15_h1: float
    event_trigger: str
    pattern_class: str
    event_type: str
    bayes_win_prob: float = 0.0
    ev_rank: float = 0.0
    ev_lot_multiplier: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in TTM_FEATURE_COLUMNS if k not in _META_COLUMNS}


_META_COLUMNS = frozenset(
    {
        "trade_id",
        "timestamp",
        "pair",
        "direction",
        "decision_source",
        "executed",
        "trade_result",
        "profit_r",
        "result_r",
        "win_loss",
        "sized_result_r",
        "shadow_result",
        "shadow_profit_r",
        "shadow_trade_result",
    }
)

TTM_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "pair",
    "direction",
    "event_type",
    "event_trigger",
    "pattern_class",
    "weekday",
    "month",
    "quarter",
    "week_of_month",
    "is_gotobi",
    "is_month_end",
    "is_quarter_end",
    "minutes_to_ttm",
    "minutes_after_ttm",
    "pre_ttm_return",
    "pre_ttm_velocity",
    "pre_ttm_range",
    "pre_ttm_atr_ratio",
    "asian_range",
    "asian_low_distance",
    "asian_range_pct",
    "low_break_distance",
    "low_break_velocity",
    "atr_m5",
    "atr_m15",
    "atr_h1",
    "atr_ratio_m5_h1",
    "atr_ratio_m15_h1",
    "hour",
    "minute",
    "minute_bucket",
    "session",
    "bayes_win_prob",
    "ev_rank",
    "ev_lot_multiplier",
    "decision_source",
    "executed",
    "trade_result",
    "win_loss",
    "profit_r",
    "result_r",
    "sized_result_r",
    "shadow_result",
    "shadow_profit_r",
    "shadow_trade_result",
)

TTM_SHORT_FEATURE_COLUMNS = TTM_FEATURE_COLUMNS


@dataclass
class TtmSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    candidate_score: float
    bar_index: int
    ttm_features: TtmFeatures
    event_trigger: EventTrigger
    pattern_class: PatternClass


def _is_gotobi(d: date) -> bool:
    return d.day in (5, 10, 15, 20, 25, 30)


def _is_month_end(d: date) -> bool:
    return (d + timedelta(days=1)).month != d.month


def _is_quarter_end(d: date) -> bool:
    return _is_month_end(d) and d.month in (3, 6, 9, 12)


def _minutes_to_ttm(jst_ts: pd.Timestamp) -> float:
    ttm_dt = jst_ts.normalize() + pd.Timedelta(
        hours=TTM_TIME_JST.hour,
        minutes=TTM_TIME_JST.minute,
    )
    return float((ttm_dt - jst_ts).total_seconds() / 60.0)


def _pip_size(pair: str) -> float:
    return pip_size_for_pair(pair) or JPY_PIP_SIZE


def build_ttm_feature_log_row(
    *,
    trade_id: str,
    setup: TtmSetup,
    trade_result: str,
    profit_r: float,
    decision_source: str = "ALLOW",
    executed: bool = True,
    shadow_result: str = "NONE",
    shadow_profit_r: float = 0.0,
    shadow_trade_result: str = "NONE",
    bayes_win_prob: float | None = None,
    ev_rank: float | None = None,
    ev_lot_multiplier: float | None = None,
) -> dict[str, Any]:
    row = setup.ttm_features.as_dict()
    win_loss = (
        "WIN"
        if trade_result == "WIN"
        else "LOSS"
        if trade_result == "LOSS"
        else trade_result
    )
    lot_mult = float(
        ev_lot_multiplier
        if ev_lot_multiplier is not None
        else setup.ttm_features.ev_lot_multiplier
    )
    sized_r = round(float(profit_r) * lot_mult, 4) if trade_result in ("WIN", "LOSS") else 0.0
    row.update(
        {
            "trade_id": trade_id,
            "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pair": setup.pair,
            "direction": setup.direction,
            "event_trigger": setup.event_trigger,
            "pattern_class": setup.pattern_class,
            "event_type": setup.ttm_features.event_type,
            "bayes_win_prob": round(
                float(bayes_win_prob if bayes_win_prob is not None else setup.ttm_features.bayes_win_prob),
                6,
            ),
            "ev_rank": round(
                float(ev_rank if ev_rank is not None else setup.ttm_features.ev_rank),
                6,
            ),
            "ev_lot_multiplier": round(lot_mult, 4),
            "decision_source": decision_source,
            "executed": executed,
            "trade_result": trade_result,
            "win_loss": win_loss,
            "profit_r": round(float(profit_r), 4),
            "result_r": round(float(profit_r), 4),
            "sized_result_r": sized_r,
            "shadow_result": shadow_result,
            "shadow_profit_r": round(float(shadow_profit_r), 4),
            "shadow_trade_result": shadow_trade_result,
        }
    )
    return {k: row.get(k, "") for k in TTM_FEATURE_COLUMNS}


def prepare_ttm_pair_data(
    usdjpy_m1_path: Path,
    *,
    usdjpy_m5_path: Path | None = None,
    usdjpy_m15_path: Path | None = None,
) -> tuple[BtOhlcvFrame, BtOhlcvFrame, BtOhlcvFrame]:
    from strategies.bt_ohlcv import resample_to_m15

    data_dir = Path(__file__).resolve().parents[1] / "data"
    m1 = clip_ttm_bt_window(BtOhlcvFrame.from_csv(usdjpy_m1_path))
    m5_p = usdjpy_m5_path or (data_dir / "usdjpy_m5_5y.csv")
    if m5_p.exists():
        m5 = clip_ttm_bt_window(BtOhlcvFrame.from_csv(m5_p))
    else:
        m5 = resample_bars_ns(m1, BAR_NS_5M)
    m15 = resample_to_m15(m1)
    return m1, m5, m15


def detect_ttm_setups_for_pair(
    df: Any,
    pair: str,
    *,
    m5_df: Any | None = None,
    m15_df: Any | None = None,
    h1_df: Any | None = None,
    input_tz: ZoneInfo | None = None,
    progress_hook: Any | None = None,
    bar_minutes: int = DEFAULT_SCAN_BAR_MINUTES,
    resume_from_bar: int | None = None,
    initial_setups: list[Any] | None = None,
    on_checkpoint: Any | None = None,
    checkpoint_every: int = 0,
    **kwargs: Any,
) -> list[TtmSetup]:
    del input_tz, h1_df, kwargs, checkpoint_every
    if pair not in ALLOWED_PAIRS:
        return []
    structure = m5_df
    m1_frame = df if isinstance(df, BtOhlcvFrame) else BtOhlcvFrame.from_arrays(as_ohlcv(df))
    total_bars = m1_frame.arrays.length

    scan_kwargs = dict(
        pip=_pip_size(pair),
        max_events_per_day=MAX_EVENTS_PER_DAY,
        bar_minutes=bar_minutes,
        progress_hook=progress_hook,
        min_bar_index=resume_from_bar,
    )
    if resume_from_bar is not None and resume_from_bar > 0 and initial_setups:
        new_setups = scan_ttm_setups_from_frames(df, structure, m15_df, pair, **scan_kwargs)
        setups = list(initial_setups) + new_setups
    else:
        setups = scan_ttm_setups_from_frames(df, structure, m15_df, pair, **scan_kwargs)

    if on_checkpoint is not None:
        on_checkpoint(total_bars, setups, None)
    return setups


class TtmStrategy(BaseStrategy):
    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        m5_df: pd.DataFrame | None = None,
        m15_df: pd.DataFrame | None = None,
    ) -> list[TtmSetup]:
        return detect_ttm_setups_for_pair(
            df,
            pair_name,
            m5_df=m5_df,
            m15_df=m15_df,
            h1_df=h1_df,
        )

    def analyze_setup(
        self,
        setup: TtmSetup,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        action = "ALLOW"
        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=setup.candidate_score,
            strategy_action=action,
            raw_features=setup.ttm_features.as_dict(),
        )
