"""

strategies/dbbs.py — Dual Bollinger Band Squeeze (DBBS): simultaneous BB20/BB50 breakout.



MTF: exec M15 / structure H1 / ATR H4. Pure BT feature-collection — no candidate_score.

"""



from __future__ import annotations



from dataclasses import dataclass

from typing import Any



import numpy as np

import pandas as pd



from strategies.base import StrategyResult

from strategies.base_strategy import BaseStrategy

from strategies.dbbs_common import (

    ALLOWED_PAIRS,

    DBBS_BAYES_PURE_PROB,

    DbbsSetupBase,

    MIN_RR_RATIO,

    SQUEEZE_MAX_HOLD_H1,

    SQUEEZE_MIN_HOLD_H1,

    STRATEGY_ABBREV,

    STRATEGY_FULL_NAME,

    build_dbbs_feature_log_row,

    build_dbbs_features_at,

    build_m15_to_h1_index,

    build_structure_state,

    breakout_simultaneous_at,

    clamp_result_r,

    compute_rr_ratio,

    day_index_from_timestamps,

    is_dbbs_enabled,

    is_dbbs_pure_data_mode,

    is_valid_stop_side,

    min_risk_distance,

    ohlcv_to_arrays,

    passes_entry_stop_take_geometry,

)

from strategies.dbbs_bear_kill_switch import get_edge_tracker, is_bear_kill_switch_enabled
from strategies.htf_trend_analyzer import clip_as_of



SETUP_TYPE = "DBBS"

SIGNAL_TYPE = "DBBS"

STRATEGY_SUB_NAME = STRATEGY_FULL_NAME





@dataclass(frozen=True)

class DbbsSetup(DbbsSetupBase):

    trail_tp_sigma: float = 1.0



    @property

    def setup_type(self) -> str:

        return SETUP_TYPE





def _squeeze_sl(

    *,

    direction: str,

    entry: float,

    open_price: float,

    bb20_middle: float,

    atr: float,

    pip: float,

) -> float:

    buffer = atr * 0.3 if atr > 0.0 else pip * 3.0

    min_risk = min_risk_distance(atr, pip)

    if direction == "BUY":

        mid_stop = bb20_middle - buffer

        structural = min(open_price, mid_stop)

        return min(structural, entry - min_risk)

    mid_stop = bb20_middle + buffer

    structural = max(open_price, mid_stop)

    return max(structural, entry + min_risk)





def _squeeze_initial_tp(direction: str, entry: float, atr: float) -> float:

    mult = 2.5 * atr if atr > 0.0 else entry * 0.001

    return entry + mult if direction == "BUY" else entry - mult





def detect_dbbs_setups(

    m15_df: pd.DataFrame,

    h1_df: pd.DataFrame,

    pair: str,

    *,

    h4_df: pd.DataFrame | None = None,

) -> list[DbbsSetup]:
    from strategies.bt_ohlcv import BtOhlcvFrame, as_ohlcv, ts_ns_to_pd

    h1_input = h1_df.to_pandas() if isinstance(h1_df, BtOhlcvFrame) else h1_df
    if h4_df is not None and isinstance(h4_df, BtOhlcvFrame):
        h4_df = h4_df.to_pandas()

    if not is_dbbs_enabled():

        return []

    if pair not in ALLOWED_PAIRS or m15_df is None or h1_input is None:

        return []

    m15_arr = as_ohlcv(m15_df)
    m15_open, m15_high, m15_low, m15_close = (
        m15_arr.open,
        m15_arr.high,
        m15_arr.low,
        m15_arr.close,
    )
    m15_ts = m15_arr.datetime_ns
    if len(m15_close) < 5 or len(h1_input) < 60:

        return []

    as_of = ts_ns_to_pd(int(m15_ts[-1]))

    h1_clipped = clip_as_of(h1_input, as_of)

    state = build_structure_state(h1_clipped, pair=pair, h4_df=h4_df, as_of=as_of)

    if state is None:

        return []

    h1_arr = as_ohlcv(h1_clipped)
    h1_open, h1_high, h1_low, h1_close = h1_arr.open, h1_arr.high, h1_arr.low, h1_arr.close
    h1_ts = h1_arr.datetime_ns
    h1_days = day_index_from_timestamps(h1_ts)

    setups: list[DbbsSetup] = []

    min_h1 = 55

    n_h1 = min(len(state.bb20_upper), len(h1_ts), len(h1_close))
    n_m15 = min(len(m15_close), len(m15_ts))
    m15_to_h1 = build_m15_to_h1_index(m15_df, h1_clipped)
    n_m15 = min(n_m15, len(m15_to_h1))

    for m_idx in range(1, n_m15):

        h_idx = int(m15_to_h1[m_idx])

        if h_idx < min_h1 or h_idx >= n_h1:

            continue



        close_h1 = float(h1_close[h_idx])

        bb20_u = float(state.bb20_upper[h_idx])

        bb20_l = float(state.bb20_lower[h_idx])

        bb50_u = float(state.bb50_upper[h_idx])

        bb50_l = float(state.bb50_lower[h_idx])

        atr = float(state.atr[h_idx])



        if breakout_simultaneous_at(close_h1, bb20_u, bb20_l, bb50_u, bb50_l, 1):

            direction = "BUY"

        elif breakout_simultaneous_at(close_h1, bb20_u, bb20_l, bb50_u, bb50_l, -1):

            direction = "SELL"

        else:

            continue



        entry = float(m15_close[m_idx])

        stop = _squeeze_sl(

            direction=direction,

            entry=entry,

            open_price=float(h1_open[h_idx]),

            bb20_middle=float(state.bb20_middle[h_idx]),

            atr=atr,

            pip=state.pip_size,

        )

        take = _squeeze_initial_tp(direction, entry, atr)

        if not passes_entry_stop_take_geometry(

            direction,

            entry,

            stop,

            take,

            atr=atr,

            pip=state.pip_size,

            min_rr=MIN_RR_RATIO,

        ):

            continue



        features = build_dbbs_features_at(

            state=state,

            h1_open=h1_open,

            h1_high=h1_high,

            h1_low=h1_low,

            h1_close=h1_close,

            h1_timestamps=h1_ts,

            h1_day_index=h1_days,

            idx=h_idx,

            pair=pair,

            direction=direction,

            signal_type=SIGNAL_TYPE,

            breakout_sim=True,

        )

        rr = compute_rr_ratio(entry, stop, take)

        features.rr_ratio = rr



        setups.append(

            DbbsSetup(

                timestamp=ts_ns_to_pd(int(m15_ts[m_idx])),

                pair=pair,

                direction=direction,

                entry_price=entry,

                stop_loss=stop,

                take_profit=take,

                bar_index_h1=h_idx,

                bar_index_m15=m_idx,

                signal_type=SIGNAL_TYPE,

                bayes_features=features,

                rr_ratio=rr,

                bayes_probability=DBBS_BAYES_PURE_PROB,

                trail_tp_sigma=1.0,

            )

        )

    return setups





def evaluate_squeeze_trailing_exit(

    *,

    setup: DbbsSetup,

    h1_close: np.ndarray,

    bb20_middle: np.ndarray,

    bb20_std: np.ndarray,

    start_h1_index: int,

) -> tuple[str, float, int]:

    """

    Trail exit: close crosses inside BB20 ±1σ (BUY: below upper1σ, SELL: above lower1σ).

    Returns (outcome_label, result_r, holding_bars).

    """

    entry = setup.entry_price

    risk = abs(entry - setup.stop_loss)

    if risk <= 0.0 or not is_valid_stop_side(setup.direction, entry, setup.stop_loss):

        return "LOSS", -1.0, 0



    end = min(len(h1_close), start_h1_index + SQUEEZE_MAX_HOLD_H1 + 1)

    for j in range(start_h1_index + 1, end):

        held = j - start_h1_index

        if held < SQUEEZE_MIN_HOLD_H1:

            continue

        close_j = float(h1_close[j])

        mid = float(bb20_middle[j])

        std = float(bb20_std[j]) if j < len(bb20_std) else 0.0

        upper1 = mid + std

        lower1 = mid - std

        if setup.direction == "BUY" and close_j < upper1:

            pnl = close_j - entry

            label = "WIN" if pnl > 0 else "LOSS"

            return label, clamp_result_r(pnl / risk), held

        if setup.direction == "SELL" and close_j > lower1:

            pnl = entry - close_j

            label = "WIN" if pnl > 0 else "LOSS"

            return label, clamp_result_r(pnl / risk), held

        if setup.direction == "BUY" and close_j <= setup.stop_loss:

            return "LOSS", -1.0, held

        if setup.direction == "SELL" and close_j >= setup.stop_loss:

            return "LOSS", -1.0, held

    return "LOSS", -1.0, min(SQUEEZE_MAX_HOLD_H1, end - start_h1_index - 1)





class DbbsStrategy(BaseStrategy):

    """Dual Bollinger Band Squeeze — simultaneous BB20/BB50 breakout with H1 trailing exit."""



    @property

    def setup_type(self) -> str:

        return SETUP_TYPE



    def detect_setups(

        self,

        df: pd.DataFrame,

        pair_name: str,

        h1_df: pd.DataFrame | None = None,

        h4_df: pd.DataFrame | None = None,

    ) -> list[DbbsSetup]:

        if h1_df is None:

            return []

        return detect_dbbs_setups(df, h1_df, pair_name, h4_df=h4_df)



    def analyze_setup(

        self,

        setup: Any,

        gbp_setup: Any | None,

        eur_setup: Any | None,

        h1_gbp: pd.DataFrame,

        h1_eur: pd.DataFrame,

    ) -> StrategyResult:

        if not isinstance(setup, DbbsSetup):

            return StrategyResult(

                is_setup=False,

                setup_type=self.setup_type,

                direction="FLAT",

                strategy_action="REJECT",

            )

        raw = setup.bayes_features.as_dict()

        raw["signal_type"] = SIGNAL_TYPE

        raw["trail_min_hold_h1"] = SQUEEZE_MIN_HOLD_H1

        raw["trail_max_hold_h1"] = SQUEEZE_MAX_HOLD_H1

        tracker = get_edge_tracker()

        edge = tracker.pre_trade_snapshot()

        raw["last_3_avg_r"] = edge["last_3_avg_r"]

        raw["edge_risk_mult"] = edge["edge_risk_mult"]

        raw["bear_kill_switch_active"] = edge["bear_kill_switch_active"]
        raw.update(
            {
                "smt_intensity": 0.0,
                "smt_diff": 0.0,
                "smt_leader": "NONE",
                "has_bos": False,
                "both_sweep": False,
                "atr_ratio": float(raw.get("bb20_width_atr_ratio", 1.0) or 1.0),
                "htf_trend_direction": "NEUTRAL",
                "htf_counter_trend": False,
                "htf_lot_multiplier": 1.0,
            }
        )

        if tracker.is_kill_active():

            return StrategyResult(

                is_setup=True,

                setup_type=self.setup_type,

                direction=setup.direction,

                entry_price=setup.entry_price,

                stop_loss=setup.stop_loss,

                take_profit=setup.take_profit,

                candidate_score=0.0,

                raw_features=raw,

                strategy_action="REJECT",

                base_risk_pct=0.0,

            )

        return StrategyResult(

            is_setup=True,

            setup_type=self.setup_type,

            direction=setup.direction,

            entry_price=setup.entry_price,

            stop_loss=setup.stop_loss,

            take_profit=setup.take_profit,

            candidate_score=0.0,

            raw_features=raw,

            strategy_action="ALLOW",

            base_risk_pct=None,

        )





__all__ = [

    "DbbsSetup",

    "DbbsStrategy",

    "SETUP_TYPE",

    "SIGNAL_TYPE",

    "STRATEGY_ABBREV",

    "STRATEGY_FULL_NAME",

    "STRATEGY_SUB_NAME",

    "build_dbbs_feature_log_row",

    "detect_dbbs_setups",

    "evaluate_squeeze_trailing_exit",

    "is_dbbs_enabled",

]


