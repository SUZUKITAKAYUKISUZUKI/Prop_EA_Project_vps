"""
strategies/dinapoli.py — DiNapoli (DN): exec M15 / structure H1 / ATR H4.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.dinapoli_universe_fast import (
    SIGNAL_BUY,
    SIGNAL_SELL,
    DiNapoliUniverseFast,
    FiboStructure,
)
from strategies.market_utils import pip_size_for_pair

logger = logging.getLogger(__name__)

SETUP_TYPE = "DINAPOLI_STRUCTURE"
STRATEGY_ABBREV = "DN"
STRATEGY_FULL_NAME = "DiNapoli"
DINAPOLI_PAIR_PRIMARY = "EURUSD"
DINAPOLI_PAIR_SECONDARY = "GBPUSD"
ALLOWED_PAIRS = frozenset({DINAPOLI_PAIR_PRIMARY, DINAPOLI_PAIR_SECONDARY})
DINAPOLI_EXEC_BAR_MINUTES = 15
DINAPOLI_STRUCTURE_BAR_MINUTES = 60
DINAPOLI_ATR_BAR_MINUTES = 240
Direction = Literal["BUY", "SELL"]
SignalKind = Literal["SINGLE_PENETRATION", "DOUBLE_REPO"]


def is_dinapoli_pure_bt_mode() -> bool:
    """Bayes 特徴量収集用: L0-L6 防御 + L4 Gemini を無効化するピュア BT。"""
    return os.getenv("DINAPOLI_PURE_BT", "0").strip().lower() in ("1", "true", "yes", "on")


def is_dinapoli_defense_mode() -> bool:
    """
    DN canonical production (default when not pure BT).

    - L0-L2 / L4.5-L6: ON
    - L3.5 generic BayesEngine: OFF (DN Prop Gate replaces it)
    - L4 Gemini: OFF
    - 3 safety brakes: ON (Profit Cushion / Twin Brake / DD Throttling)
    """
    if is_dinapoli_pure_bt_mode():
        return False
    flag = os.getenv("DINAPOLI_DEFENSE", "1").strip().lower()
    return flag in ("1", "true", "yes", "on")


def is_dinapoli_defense_pure_mode() -> bool:
    """DiNapoli 防御レイヤー純粋モード（``DINAPOLI_PURE_BT=1``）。"""
    return is_dinapoli_pure_bt_mode()


def is_dinapoli_l4_bypass() -> bool:
    """DiNapoli: Gemini L4 監査は常に無効（pure BT / canonical defense）。"""
    return is_dinapoli_pure_bt_mode() or is_dinapoli_defense_mode()


def is_dinapoli_generic_bayes_bypass() -> bool:
    """Generic L3.5 BayesEngine をスキップ — DN Prop Gate が L3.5 代替。"""
    return is_dinapoli_pure_bt_mode() or is_dinapoli_defense_mode()


def configure_dinapoli_defense_env() -> None:
    """Apply DN canonical defense defaults (idempotent)."""
    if is_dinapoli_pure_bt_mode():
        return
    os.environ.setdefault("DINAPOLI_DEFENSE", "1")
    os.environ.setdefault("DN_PROP_GATE", "1")
    os.environ.setdefault("CHALLENGE_BASE_RISK_PCT_MAX", "0.006")
    os.environ.setdefault("DN_PROP_GATE_BASE_RISK_PCT", "0.006")
    os.environ.setdefault("PROFIT_CUSHION_ENABLED", "1")
    os.environ.setdefault("TWIN_BRAKE_ENABLED", "1")
    os.environ.setdefault("DD_THROTTLING_ENABLED", "1")


@dataclass(frozen=True)
class DiNapoliSetup:
    timestamp: pd.Timestamp
    bar_index: int
    pair: str
    direction: Direction
    signal: int
    entry_price: float
    stop_loss: float
    take_profit: float
    dma_3_3: float
    macd_line: float
    macd_signal: float
    stochastics: float
    cop: float
    op: float
    xop: float
    retracement: float
    a_idx: int
    b_idx: int
    c_idx: int

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE


def _nearest_structure(
    structures: list[FiboStructure],
    bar_index: int,
    direction: int,
) -> FiboStructure | None:
    best: FiboStructure | None = None
    best_dist = 10**9
    for st in structures:
        if st.direction != direction:
            continue
        if st.c_idx > bar_index:
            continue
        dist = bar_index - st.c_idx
        if dist < best_dist:
            best_dist = dist
            best = st
    return best


def _build_setup_from_signal(
    *,
    bar_index: int,
    timestamp: pd.Timestamp,
    pair: str,
    direction: Direction,
    signal: int,
    close: float,
    atr_proxy: float,
    indicators: dict[str, np.ndarray],
    structures: list[FiboStructure],
) -> DiNapoliSetup:
    dir_int = 1 if direction == "BUY" else -1
    st = _nearest_structure(structures, bar_index, dir_int)
    pip = pip_size_for_pair(pair)
    risk = max(atr_proxy, 15.0 * pip)
    if direction == "BUY":
        stop_loss = close - risk
        take_profit = close + 2.0 * risk
        if st is not None:
            take_profit = st.op
    else:
        stop_loss = close + risk
        take_profit = close - 2.0 * risk
        if st is not None:
            take_profit = st.op

    return DiNapoliSetup(
        timestamp=timestamp,
        bar_index=bar_index,
        pair=pair,
        direction=direction,
        signal=signal,
        entry_price=close,
        stop_loss=stop_loss,
        take_profit=take_profit,
        dma_3_3=float(indicators["dma_3_3"][bar_index]),
        macd_line=float(indicators["macd_line"][bar_index]),
        macd_signal=float(indicators["macd_signal"][bar_index]),
        stochastics=float(indicators["stochastics"][bar_index]),
        cop=float(st.cop) if st else close,
        op=float(st.op) if st else take_profit,
        xop=float(st.xop) if st else take_profit,
        retracement=float(st.retracement) if st else 0.0,
        a_idx=int(st.a_idx) if st else -1,
        b_idx=int(st.b_idx) if st else -1,
        c_idx=int(st.c_idx) if st else -1,
    )


def detect_dinapoli_setups_from_arrays(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray,
    pair: str,
    engine: DiNapoliUniverseFast | None = None,
) -> list[DiNapoliSetup]:
    engine = engine or DiNapoliUniverseFast()
    indicators, structures, signals = engine.run_pipeline(high, low, close, timestamps)
    n = close.shape[0]
    atr_proxy = 0.0
    if n >= 15:
        tr_sum = 0.0
        for j in range(n - 14, n):
            tr = high[j] - low[j]
            if tr > atr_proxy:
                atr_proxy = tr
    setups: list[DiNapoliSetup] = []
    for i in range(n):
        sig = int(signals[i])
        if sig == SIGNAL_BUY:
            direction: Direction = "BUY"
        elif sig == SIGNAL_SELL:
            direction = "SELL"
        else:
            continue
        ts_val = timestamps[i]
        if isinstance(ts_val, (np.datetime64,)):
            ts = pd.Timestamp(ts_val)
        elif isinstance(ts_val, (int, np.integer)):
            ts = pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=int(ts_val))
        else:
            ts = pd.Timestamp(ts_val)
        setups.append(
            _build_setup_from_signal(
                bar_index=i,
                timestamp=ts,
                pair=pair,
                direction=direction,
                signal=sig,
                close=float(close[i]),
                atr_proxy=atr_proxy,
                indicators=indicators,
                structures=structures,
            )
        )
    return setups


def compute_dinapoli_candidate_score(setup: DiNapoliSetup) -> tuple[float, dict[str, float]]:
    """
    COP/OP/XOP 達成可能性と A-B-C 構造品質に基づく candidate_score。

    - retracement: 0.382–0.618 帯、0.500 が理想
    - rr_to_op: entry→OP / entry→SL（設計上 OP=TP）
    - stochastics: BUY≤20 / SELL≥80 が理想
    """
    retrace_quality = max(0.0, 1.0 - abs(setup.retracement - 0.500) * 4.0)

    sl_distance = abs(setup.stop_loss - setup.entry_price)
    if sl_distance > 0.0:
        rr_to_op = abs(setup.op - setup.entry_price) / sl_distance
        rr_score = min(1.0, rr_to_op / 3.0)
    else:
        rr_to_op = 0.0
        rr_score = 0.0

    if setup.direction == "BUY":
        stoch_score = max(0.0, 1.0 - setup.stochastics / 30.0)
    else:
        stoch_score = max(0.0, (setup.stochastics - 70.0) / 30.0)

    score = 40.0 + retrace_quality * 25.0 + rr_score * 20.0 + stoch_score * 15.0
    components = {
        "retrace_quality": retrace_quality,
        "rr_to_op": rr_to_op,
        "rr_score": rr_score,
        "stoch_score": stoch_score,
    }
    return score, components


class DiNapoliStrategy(BaseStrategy):
    """DiNapoli A-B-C structure strategy using DiNapoliUniverseFast engine."""

    def __init__(
        self,
        weights_config: dict[str, int] | None = None,
        mode_h1: bool = False,
        *,
        zigzag_dev: float = 0.001,
        min_thrust_bars: int = 8,
    ):
        super().__init__(weights_config, mode_h1)
        self.engine = DiNapoliUniverseFast(
            zigzag_dev=zigzag_dev,
            min_thrust_bars=min_thrust_bars,
        )

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        h4_df: pd.DataFrame | None = None,
    ) -> list[DiNapoliSetup]:
        if pair_name not in ALLOWED_PAIRS:
            return []
        if df is None or df.empty:
            return []
        from strategies.dinapoli_mtf import detect_dinapoli_setups_for_pair

        return detect_dinapoli_setups_for_pair(
            df,
            pair_name,
            h1_df=h1_df,
            h4_df=h4_df,
            engine=self.engine,
        )

    def analyze_setup(
        self,
        setup: Any,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        del gbp_setup, eur_setup, h1_gbp, h1_eur
        if not isinstance(setup, DiNapoliSetup):
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
            )
        score, score_parts = compute_dinapoli_candidate_score(setup)
        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            strategy_action="ACCEPT",
            candidate_score=score,
            raw_features={
                "signal": setup.signal,
                "retracement": setup.retracement,
                "cop": setup.cop,
                "op": setup.op,
                "xop": setup.xop,
                "dma_3_3": setup.dma_3_3,
                "macd_line": setup.macd_line,
                "macd_signal": setup.macd_signal,
                "stochastics": setup.stochastics,
                **score_parts,
                "smt_intensity": 0.0,
                "smt_diff": 0.0,
                "smt_leader": "NONE",
                "has_bos": False,
                "both_sweep": False,
                "atr_ratio": 1.0,
                "htf_trend_direction": "NEUTRAL",
                "htf_counter_trend": False,
                "htf_lot_multiplier": 1.0,
            },
        )


__all__ = [
    "ALLOWED_PAIRS",
    "DINAPOLI_ATR_BAR_MINUTES",
    "DINAPOLI_EXEC_BAR_MINUTES",
    "DINAPOLI_PAIR_PRIMARY",
    "DINAPOLI_PAIR_SECONDARY",
    "DINAPOLI_STRUCTURE_BAR_MINUTES",
    "DiNapoliSetup",
    "DiNapoliStrategy",
    "compute_dinapoli_candidate_score",
    "configure_dinapoli_defense_env",
    "is_dinapoli_defense_mode",
    "is_dinapoli_defense_pure_mode",
    "is_dinapoli_generic_bayes_bypass",
    "is_dinapoli_l4_bypass",
    "is_dinapoli_pure_bt_mode",
    "SETUP_TYPE",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "detect_dinapoli_setups_from_arrays",
]
