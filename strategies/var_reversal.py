"""
strategies/var_reversal.py — Volume Area Reversal (VAR): Phase0 Pure Discovery.

Regime → VP → PA → Entry. No candidate_score / Bayes / ML / optimization in pure mode.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import clip_as_of
from strategies.market_utils import pip_size_for_pair
from strategies.var_detector import (
    ALLOWED_PAIRS,
    EXEC_BAR_MINUTES,
    EXIT_TP_POC,
    EXIT_TP_VA,
    SESSION_START_HOUR_UTC,
    STRUCTURE_BAR_MINUTES,
    VAR_PURE_DATA_MODE,
    WARMUP_BARS,
    compute_atr_series,
    compute_bb_width_series,
    compute_hv_series,
    exit_code_to_str,
    is_var_enabled,
    is_var_pure_data_mode,
    pa_flags_to_str,
    precompute_entropy_series,
    precompute_hurst_series,
    scan_var_events_numba,
)
from volume_profile_analyzer import SessionVolumeProfile

SETUP_TYPE = "VAR"
STRATEGY_ABBREV = "VAR"
STRATEGY_FULL_NAME = "Volume Area Reversal"
VAR_BAYES_PURE_PROB = 1.0

VAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "pair_name",
    "pair",
    "direction",
    "entry_price",
    "exit_price",
    "stop_loss",
    "take_profit",
    "tp_target",
    "vah",
    "val",
    "poc",
    "vp_touch_side",
    "pa_types",
    "atr_h1",
    "atr_percentile_20d",
    "atr_percentile_100b",
    "atr_vs_session_avg",
    "bb20_width_pips",
    "bb20_width_percentile",
    "bb50_width_pips",
    "bb50_width_percentile",
    "hv_20",
    "hv_percentile",
    "session_range_pips",
    "session_range_atr_ratio",
    "recent_5bar_range_pips",
    "recent_5bar_range_atr_ratio",
    "atr_change_rate",
    "bb_width_change_rate",
    "feat_entropy_20",
    "feat_entropy_50",
    "feat_entropy_100",
    "entropy_percentile_100",
    "entropy_change_5",
    "entropy_change_10",
    "feat_hurst_50",
    "feat_hurst_100",
    "time_to_exit_minutes",
    "bars_held",
    "exit_reason",
    "volatility_expansion_detected",
    "result_r",
    "profit_r",
    "setup_type",
    "executed",
    "trade_result",
    "bayes_probability",
    "decision_source",
    "bar_index",
)


@dataclass(frozen=True)
class VarSetup:
    timestamp: pd.Timestamp
    pair_name: str
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    tp_target: str
    vah: float
    val: float
    poc: float
    vp_touch_side: str
    pa_types: str
    bar_index: int
    atr_h1: float
    atr_percentile_20d: float
    atr_percentile_100b: float
    atr_vs_session_avg: float
    bb20_width_pips: float
    bb20_width_percentile: float
    bb50_width_pips: float
    bb50_width_percentile: float
    hv_20: float
    hv_percentile: float
    session_range_pips: float
    session_range_atr_ratio: float
    recent_5bar_range_pips: float
    recent_5bar_range_atr_ratio: float
    atr_change_rate: float
    bb_width_change_rate: float
    feat_entropy_20: float
    feat_entropy_50: float
    feat_entropy_100: float
    entropy_percentile_100: float
    entropy_change_5: float
    entropy_change_10: float
    feat_hurst_50: float
    feat_hurst_100: float
    time_to_exit_minutes: int
    bars_held: int
    exit_reason: str
    volatility_expansion_detected: bool
    result_r: float
    profit_r: float
    setup_type: str = SETUP_TYPE

    @property
    def bayes_features(self) -> VarSetup:
        return self

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _to_ns(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values("datetime").reset_index(drop=True)


def _frame_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    work = _prepare_frame(df)
    vol = work["volume"].astype(np.float64).to_numpy() if "volume" in work.columns else np.ones(len(work))
    return {
        "dt_ns": _to_ns(work["datetime"]),
        "open": work["open"].astype(np.float64).to_numpy(),
        "high": work["high"].astype(np.float64).to_numpy(),
        "low": work["low"].astype(np.float64).to_numpy(),
        "close": work["close"].astype(np.float64).to_numpy(),
        "volume": vol,
        "datetime": work["datetime"],
    }


def _tp_target_label(tp_kind: int) -> str:
    if int(tp_kind) == EXIT_TP_POC:
        return "POC"
    if int(tp_kind) == EXIT_TP_VA:
        return "VA"
    return "NONE"


def _build_setup_from_scan_row(
    *,
    bar_index: int,
    direction_code: int,
    pa_flags: int,
    arrays: dict[str, np.ndarray],
    row_data: dict[str, Any],
) -> VarSetup:
    direction = "SELL" if direction_code < 0 else "BUY"
    vp_touch = "VAH" if direction == "SELL" else "VAL"
    ts = pd.Timestamp(arrays["datetime"].iloc[bar_index])
    pair = str(row_data["pair"])
    result_r = float(row_data["result_r"])
    return VarSetup(
        timestamp=ts,
        pair_name=pair,
        pair=pair,
        direction=direction,
        entry_price=round(float(row_data["entry"]), 5),
        exit_price=round(float(row_data["exit_px"]), 5),
        stop_loss=round(float(row_data["sl"]), 5),
        take_profit=round(float(row_data["tp"]), 5),
        tp_target=_tp_target_label(int(row_data["tp_kind"])),
        vah=round(float(row_data["vah"]), 5),
        val=round(float(row_data["val"]), 5),
        poc=round(float(row_data["poc"]), 5),
        vp_touch_side=vp_touch,
        pa_types=pa_flags_to_str(int(pa_flags)),
        bar_index=int(bar_index),
        atr_h1=round(float(row_data["atr_h1"]), 6),
        atr_percentile_20d=round(float(row_data["atr_pct20"]), 4),
        atr_percentile_100b=round(float(row_data["atr_pct100"]), 4),
        atr_vs_session_avg=round(float(row_data["atr_sess"]), 6),
        bb20_width_pips=round(float(row_data["bb20"]), 4),
        bb20_width_percentile=round(float(row_data["bb20_pct"]), 4),
        bb50_width_pips=round(float(row_data["bb50"]), 4),
        bb50_width_percentile=round(float(row_data["bb50_pct"]), 4),
        hv_20=round(float(row_data["hv20"]), 6),
        hv_percentile=round(float(row_data["hv_pct"]), 4),
        session_range_pips=round(float(row_data["sess_rng"]), 4),
        session_range_atr_ratio=round(float(row_data["sess_rng_atr"]), 4),
        recent_5bar_range_pips=round(float(row_data["recent5_rng"]), 4),
        recent_5bar_range_atr_ratio=round(float(row_data["recent5_rng_atr"]), 4),
        atr_change_rate=round(float(row_data["atr_chg"]), 6),
        bb_width_change_rate=round(float(row_data["bb_chg"]), 6),
        feat_entropy_20=round(float(row_data["ent20"]), 6),
        feat_entropy_50=round(float(row_data["ent50"]), 6),
        feat_entropy_100=round(float(row_data["ent100"]), 6),
        entropy_percentile_100=round(float(row_data["ent_pct"]), 4),
        entropy_change_5=round(float(row_data["ent_chg5"]), 6),
        entropy_change_10=round(float(row_data["ent_chg10"]), 6),
        feat_hurst_50=round(float(row_data["h50"]), 6),
        feat_hurst_100=round(float(row_data["h100"]), 6),
        time_to_exit_minutes=int(row_data["exit_mins"]),
        bars_held=int(row_data["bars_held"]),
        exit_reason=str(row_data["exit_reason"]),
        volatility_expansion_detected=bool(row_data["vol_exp"]),
        result_r=round(result_r, 4),
        profit_r=round(result_r, 4),
    )


def detect_var_setups(
    h1_df: pd.DataFrame,
    pair: str,
    *,
    m5_df: pd.DataFrame | None = None,
    h4_df: pd.DataFrame | None = None,
    max_events: int = 50_000,
    **kwargs: Any,
) -> list[VarSetup]:
    del h4_df, kwargs
    pair_u = str(pair).upper()
    if pair_u not in ALLOWED_PAIRS:
        return []
    if not is_var_enabled():
        return []
    if m5_df is None or m5_df.empty:
        return []

    exec_arrays = _frame_arrays(h1_df)
    vp_arrays = _frame_arrays(m5_df)
    pip = pip_size_for_pair(pair_u)
    profiler = SessionVolumeProfile.for_pair(pair_u)
    bin_step = float(profiler.bin_step)

    close = exec_arrays["close"]
    atr = compute_atr_series(exec_arrays["high"], exec_arrays["low"], close, 14)
    bb20, _, _ = compute_bb_width_series(close, 20, 2.0, pip)
    bb50, _, _ = compute_bb_width_series(close, 50, 2.0, pip)
    hv = compute_hv_series(close, 20)
    ent20, ent50, ent100 = precompute_entropy_series(close)
    h50, h100 = precompute_hurst_series(close)

    scan = scan_var_events_numba(
        exec_arrays["dt_ns"],
        exec_arrays["open"],
        exec_arrays["high"],
        exec_arrays["low"],
        close,
        exec_arrays["volume"],
        vp_arrays["dt_ns"],
        vp_arrays["open"],
        vp_arrays["high"],
        vp_arrays["low"],
        vp_arrays["close"],
        vp_arrays["volume"],
        atr,
        bb20,
        bb50,
        hv,
        ent20,
        ent50,
        ent100,
        h50,
        h100,
        pip,
        bin_step,
        WARMUP_BARS,
        max_events,
    )
    (
        out_idx,
        out_dir,
        out_pa,
        out_vah,
        out_val,
        out_poc,
        out_entry,
        out_sl,
        out_tp,
        out_tp_kind,
        out_atr,
        out_atr_pct20,
        out_atr_pct100,
        out_atr_sess,
        out_bb20,
        out_bb20_pct,
        out_bb50,
        out_bb50_pct,
        out_hv20,
        out_hv_pct,
        out_sess_rng,
        out_sess_rng_atr,
        out_recent5_rng,
        out_recent5_rng_atr,
        out_atr_chg,
        out_bb_chg,
        out_ent20,
        out_ent50,
        out_ent100,
        out_ent_pct,
        out_ent_chg5,
        out_ent_chg10,
        out_h50,
        out_h100,
        out_vol_exp,
        out_exit_code,
        out_exit_px,
        out_exit_mins,
        out_bars_held,
        out_result_r,
    ) = scan

    setups: list[VarSetup] = []
    for i in range(out_idx.shape[0]):
        row = {
            "pair": pair_u,
            "entry": out_entry[i],
            "sl": out_sl[i],
            "tp": out_tp[i],
            "tp_kind": out_tp_kind[i],
            "vah": out_vah[i],
            "val": out_val[i],
            "poc": out_poc[i],
            "atr_h1": out_atr[i],
            "atr_pct20": out_atr_pct20[i],
            "atr_pct100": out_atr_pct100[i],
            "atr_sess": out_atr_sess[i],
            "bb20": out_bb20[i],
            "bb20_pct": out_bb20_pct[i],
            "bb50": out_bb50[i],
            "bb50_pct": out_bb50_pct[i],
            "hv20": out_hv20[i],
            "hv_pct": out_hv_pct[i],
            "sess_rng": out_sess_rng[i],
            "sess_rng_atr": out_sess_rng_atr[i],
            "recent5_rng": out_recent5_rng[i],
            "recent5_rng_atr": out_recent5_rng_atr[i],
            "atr_chg": out_atr_chg[i],
            "bb_chg": out_bb_chg[i],
            "ent20": out_ent20[i],
            "ent50": out_ent50[i],
            "ent100": out_ent100[i],
            "ent_pct": out_ent_pct[i],
            "ent_chg5": out_ent_chg5[i],
            "ent_chg10": out_ent_chg10[i],
            "h50": out_h50[i],
            "h100": out_h100[i],
            "vol_exp": out_vol_exp[i],
            "exit_px": out_exit_px[i],
            "exit_mins": out_exit_mins[i],
            "bars_held": out_bars_held[i],
            "exit_reason": exit_code_to_str(int(out_exit_code[i])),
            "result_r": out_result_r[i],
        }
        setups.append(
            _build_setup_from_scan_row(
                bar_index=int(out_idx[i]),
                direction_code=int(out_dir[i]),
                pa_flags=int(out_pa[i]),
                arrays=exec_arrays,
                row_data=row,
            )
        )
    return setups


def compute_session_vp_levels(
    m5_df: pd.DataFrame,
    pair: str,
    as_of: pd.Timestamp,
) -> dict[str, float]:
    """Session VP with clip_as_of — no future data."""
    clipped = clip_as_of(_prepare_frame(m5_df), as_of)
    if clipped.empty:
        return {"vah": float("nan"), "val": float("nan"), "poc": float("nan")}
    ts_ns = _to_ns(clipped["datetime"])[-1]
    sess_start = pd.Timestamp(session_start_ns_from_ts(int(ts_ns)), unit="ns")
    sess = clipped[pd.to_datetime(clipped["datetime"]) >= sess_start]
    profile = SessionVolumeProfile.for_pair(pair).calculate_profile(sess)
    return profile


def session_start_ns_from_ts(ts_ns: int) -> int:
    ns_per_day = 86_400_000_000_000
    ns_per_hour = 3_600_000_000_000
    day_start = (ts_ns // ns_per_day) * ns_per_day
    start = day_start + SESSION_START_HOUR_UTC * ns_per_hour
    if ts_ns < start:
        start -= ns_per_day
    return start


def build_var_feature_log_row(setup: VarSetup, trade_id: str = "") -> dict[str, Any]:
    trade_result = "WIN" if setup.result_r > 0 else "LOSS"
    return {
        "trade_id": trade_id,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "pair_name": setup.pair_name,
        "pair": setup.pair,
        "direction": setup.direction,
        "entry_price": setup.entry_price,
        "exit_price": setup.exit_price,
        "stop_loss": setup.stop_loss,
        "take_profit": setup.take_profit,
        "tp_target": setup.tp_target,
        "vah": setup.vah,
        "val": setup.val,
        "poc": setup.poc,
        "vp_touch_side": setup.vp_touch_side,
        "pa_types": setup.pa_types,
        "atr_h1": setup.atr_h1,
        "atr_percentile_20d": setup.atr_percentile_20d,
        "atr_percentile_100b": setup.atr_percentile_100b,
        "atr_vs_session_avg": setup.atr_vs_session_avg,
        "bb20_width_pips": setup.bb20_width_pips,
        "bb20_width_percentile": setup.bb20_width_percentile,
        "bb50_width_pips": setup.bb50_width_pips,
        "bb50_width_percentile": setup.bb50_width_percentile,
        "hv_20": setup.hv_20,
        "hv_percentile": setup.hv_percentile,
        "session_range_pips": setup.session_range_pips,
        "session_range_atr_ratio": setup.session_range_atr_ratio,
        "recent_5bar_range_pips": setup.recent_5bar_range_pips,
        "recent_5bar_range_atr_ratio": setup.recent_5bar_range_atr_ratio,
        "atr_change_rate": setup.atr_change_rate,
        "bb_width_change_rate": setup.bb_width_change_rate,
        "feat_entropy_20": setup.feat_entropy_20,
        "feat_entropy_50": setup.feat_entropy_50,
        "feat_entropy_100": setup.feat_entropy_100,
        "entropy_percentile_100": setup.entropy_percentile_100,
        "entropy_change_5": setup.entropy_change_5,
        "entropy_change_10": setup.entropy_change_10,
        "feat_hurst_50": setup.feat_hurst_50,
        "feat_hurst_100": setup.feat_hurst_100,
        "time_to_exit_minutes": setup.time_to_exit_minutes,
        "bars_held": setup.bars_held,
        "exit_reason": setup.exit_reason,
        "volatility_expansion_detected": setup.volatility_expansion_detected,
        "result_r": setup.result_r,
        "profit_r": setup.profit_r,
        "setup_type": setup.setup_type,
        "executed": True,
        "trade_result": trade_result,
        "bayes_probability": VAR_BAYES_PURE_PROB,
        "decision_source": "ALLOW",
        "bar_index": setup.bar_index,
    }


class VarStrategy(BaseStrategy):
    """VAR Phase0 — pure discovery, no scoring filters."""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> list[VarSetup]:
        exec_df = h1_df if h1_df is not None else df
        return detect_var_setups(
            exec_df,
            pair_name,
            m5_df=kwargs.get("m5_df"),
            h4_df=kwargs.get("h4_df"),
        )

    def analyze_setup(
        self,
        setup: VarSetup,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        del gbp_setup, eur_setup, h1_gbp, h1_eur
        raw = setup.as_dict()
        raw["bayes_probability"] = VAR_BAYES_PURE_PROB
        raw["decision_source"] = "ALLOW"
        return StrategyResult(
            is_setup=True,
            setup_type=SETUP_TYPE,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=0.0,
            raw_features=raw,
            strategy_action="ALLOW",
        )


def is_var_pure_bt_mode() -> bool:
    return is_var_pure_data_mode()


__all__ = [
    "ALLOWED_PAIRS",
    "EXEC_BAR_MINUTES",
    "SETUP_TYPE",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "STRUCTURE_BAR_MINUTES",
    "VAR_FEATURE_COLUMNS",
    "VAR_PURE_DATA_MODE",
    "VarSetup",
    "VarStrategy",
    "build_var_feature_log_row",
    "compute_session_vp_levels",
    "detect_var_setups",
    "is_var_enabled",
    "is_var_pure_bt_mode",
    "is_var_pure_data_mode",
]
