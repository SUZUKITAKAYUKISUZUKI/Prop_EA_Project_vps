"""DiNapoli entry-time feature vector + exit outcome patch for dn_feature_log."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from audit import risk_manager as audit_rm
from storage.dn_feature_store import DN_FEATURE_COLUMNS
from strategies.bt_ohlcv import as_ohlcv, ts_ns_to_pd
from strategies.archive.cspa import resolve_cspa_session_type
from strategies.archive.cspa_arrays import compute_atr_np
from src.filters.dn_bayes_ev import apply_dn_ev_to_row, is_dn_ev_rank_enabled
from src.filters.dn_prop_gate_v1 import is_dn_prop_gate_enabled
from strategies.dinapoli import DiNapoliSetup, compute_dinapoli_candidate_score
from strategies.dinapoli_mtf import map_htf_index
from strategies.dinapoli_universe_fast import DiNapoliUniverseFast
from strategies.market_utils import pip_size_for_pair

LONDON_OPEN_HOUR = 8
NY_OPEN_HOUR = 13


def _is_gotobi(d: date) -> bool:
    return d.day in (5, 10, 15, 20, 25, 30)


def _ema_last(closes: np.ndarray, period: int) -> float:
    if closes.size < period:
        return float("nan")
    alpha = 2.0 / (period + 1.0)
    ema = float(closes[0])
    for px in closes[1:]:
        ema = alpha * float(px) + (1.0 - alpha) * ema
    return ema


def _ema_slope(closes: np.ndarray, period: int, lookback: int = 3) -> float:
    if closes.size < period + lookback:
        return 0.0
    end = _ema_last(closes, period)
    start = _ema_last(closes[:-lookback], period)
    return float(end - start)


def _rsi_last(closes: np.ndarray, period: int = 14) -> float:
    if closes.size < period + 1:
        return float("nan")
    delta = np.diff(closes[-(period + 1) :])
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss <= 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if close.size < period + 1:
        return float("nan")
    atr = compute_atr_np(high, low, close, period)
    val = float(atr[-1])
    return val if not np.isnan(val) else float("nan")


def _session_extremes(
    dt_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bar_index: int,
    session: str,
) -> tuple[float, float]:
    ts = ts_ns_to_pd(int(dt_ns[bar_index]))
    day = ts.normalize()
    mask = np.zeros(bar_index + 1, dtype=bool)
    hours = pd.to_datetime(dt_ns[: bar_index + 1].astype("datetime64[ns]")).hour
    if session == "ASIA":
        sel = (hours >= 0) & (hours < 8)
    elif session == "LONDON":
        sel = (hours >= 8) & (hours < 13)
    else:
        sel = np.ones(bar_index + 1, dtype=bool)
    day_vals = pd.to_datetime(dt_ns[: bar_index + 1].astype("datetime64[ns]")).normalize()
    day_mask = np.asarray(day_vals == day, dtype=bool)
    mask = sel & day_mask
    if not mask.any():
        mask = day_mask
    return float(high[: bar_index + 1][mask].max()), float(low[: bar_index + 1][mask].min())


def _daily_extremes(
    dt_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bar_index: int,
) -> tuple[float, float]:
    day = ts_ns_to_pd(int(dt_ns[bar_index])).normalize()
    days = pd.to_datetime(dt_ns[: bar_index + 1].astype("datetime64[ns]")).normalize()
    mask = np.asarray(days == day, dtype=bool)
    return float(high[: bar_index + 1][mask].max()), float(low[: bar_index + 1][mask].min())


def _volume_zscore(volume: np.ndarray, bar_index: int, window: int = 48) -> float:
    start = max(0, bar_index - window + 1)
    chunk = volume[start : bar_index + 1]
    if chunk.size < 5:
        return 0.0
    mu = float(chunk.mean())
    sd = float(chunk.std())
    if sd <= 0:
        return 0.0
    return float((volume[bar_index] - mu) / sd)


def _volatility_regime(atr_series: np.ndarray, bar_index: int, window: int = 96) -> float:
    start = max(0, bar_index - window + 1)
    chunk = atr_series[start : bar_index + 1]
    chunk = chunk[~np.isnan(chunk)]
    if chunk.size < 5 or bar_index >= atr_series.size:
        return 0.5
    cur = float(atr_series[bar_index])
    if np.isnan(cur):
        return 0.5
    return float(np.mean(chunk <= cur))


def _resolve_llm_decision(decision_source: str, confidence: int) -> str:
    if decision_source == "REJECT_BY_LLM":
        return "REJECT_BY_LLM"
    if decision_source == "CAUTION":
        return "CAUTION"
    if decision_source == "ALLOW":
        return "ALLOW"
    if confidence > 0:
        return audit_rm.confidence_to_llm_decision(confidence)
    return decision_source or "NONE"


def compute_excursions_r(
    pair_df: Any,
    start_index: int,
    direction: str,
    entry: float,
    stop_loss: float,
    *,
    max_holding_bars: int,
) -> tuple[float, float, int]:
    ohlcv = as_ohlcv(pair_df)
    base_risk = abs(entry - stop_loss)
    if base_risk <= 0 or start_index < 0:
        return 0.0, 0.0, 0
    end_index = min(start_index + max_holding_bars, ohlcv.length - 1)
    mfe = 0.0
    mae = 0.0
    for i in range(start_index + 1, end_index + 1):
        hi = float(ohlcv.high[i])
        lo = float(ohlcv.low[i])
        if direction == "BUY":
            mfe = max(mfe, (hi - entry) / base_risk)
            mae = max(mae, (entry - lo) / base_risk)
        else:
            mfe = max(mfe, (entry - lo) / base_risk)
            mae = max(mae, (hi - entry) / base_risk)
    holding = max(0, end_index - start_index) * 15
    return round(mfe, 4), round(mae, 4), holding


@dataclass(frozen=True)
class DnFeatureLogRuntime:
    store: Any
    run_id: str = "bt"
    wft_window: int = -1
    oos_start: pd.Timestamp | None = None
    oos_end: pd.Timestamp | None = None


@dataclass(frozen=True)
class DnFeatureMarketContext:
    exec_m15: Any
    h1: Any | None = None
    h4: Any | None = None


def build_dn_entry_features(
    *,
    setup: DiNapoliSetup,
    trade_id: str,
    run_id: str,
    decision_source: str,
    llm_confidence: int,
    llm_reason: str,
    minutes_to_news: int,
    ctx: DnFeatureMarketContext,
    wft_window: int = -1,
    is_oos: int = 0,
) -> dict[str, Any]:
    ohlcv = as_ohlcv(ctx.exec_m15)
    idx = int(setup.bar_index)
    idx = min(max(idx, 0), ohlcv.length - 1)
    pip = pip_size_for_pair(setup.pair)
    ts = pd.Timestamp(setup.timestamp)
    session = resolve_cspa_session_type(ts)
    closes = ohlcv.close[: idx + 1]
    highs = ohlcv.high[: idx + 1]
    lows = ohlcv.low[: idx + 1]

    ema20 = _ema_last(closes, 20)
    ema50 = _ema_last(closes, 50)
    ema200 = _ema_last(closes, 200)
    if setup.direction == "BUY":
        align = int(ema20 >= ema50 >= ema200)
    else:
        align = int(ema20 <= ema50 <= ema200)

    engine = DiNapoliUniverseFast()
    ind = engine.compute_all_indicators(ohlcv.high, ohlcv.low, ohlcv.close)
    atr_m15 = _atr_last(highs, lows, closes, 14)
    atr_m15_arr = compute_atr_np(ohlcv.high, ohlcv.low, ohlcv.close, 14)

    atr_h1 = float("nan")
    atr_h4 = float("nan")
    rsi_h1 = float("nan")
    h1_macd_spread = 0.0
    if ctx.h1 is not None:
        h1 = as_ohlcv(ctx.h1)
        h1_idx = int(map_htf_index(np.array([ohlcv.datetime_ns[idx]]), h1.datetime_ns)[0])
        if h1_idx >= 0:
            h1_cl = h1.close[: h1_idx + 1]
            h1_hi = h1.high[: h1_idx + 1]
            h1_lo = h1.low[: h1_idx + 1]
            atr_h1 = _atr_last(h1_hi, h1_lo, h1_cl, 14)
            rsi_h1 = _rsi_last(h1_cl, 14)
            h1_ind = engine.compute_all_indicators(h1.high, h1.low, h1.close)
            h1_macd_spread = float(h1_ind["macd_line"][h1_idx] - h1_ind["macd_signal"][h1_idx])

    if ctx.h4 is not None:
        h4 = as_ohlcv(ctx.h4)
        h4_idx = int(map_htf_index(np.array([ohlcv.datetime_ns[idx]]), h4.datetime_ns)[0])
        if h4_idx >= 0:
            atr_h4 = _atr_last(
                h4.high[: h4_idx + 1],
                h4.low[: h4_idx + 1],
                h4.close[: h4_idx + 1],
                14,
            )

    if np.isnan(atr_h4) or atr_h4 <= 0:
        atr_h4 = abs(setup.entry_price - setup.stop_loss)

    asia_hi, asia_lo = _session_extremes(ohlcv.datetime_ns, ohlcv.high, ohlcv.low, idx, "ASIA")
    london_hi, london_lo = _session_extremes(ohlcv.datetime_ns, ohlcv.high, ohlcv.low, idx, "LONDON")
    day_hi, day_lo = _daily_extremes(ohlcv.datetime_ns, ohlcv.high, ohlcv.low, idx)

    def _dist_pips(level: float) -> float:
        return abs(setup.entry_price - level) / pip if pip > 0 else 0.0

    swing_size = 0.0
    swing_duration = 0.0
    if setup.a_idx >= 0 and setup.b_idx >= 0:
        swing_duration = float(max(0, setup.b_idx - setup.a_idx))
        if setup.b_idx < len(closes) and setup.a_idx < len(closes):
            swing_size = abs(float(closes[setup.b_idx]) - float(closes[setup.a_idx])) / pip

    fib_distance = abs(setup.retracement - 0.5) * 100.0
    score, _score_parts = compute_dinapoli_candidate_score(setup)
    stoch = float(setup.stochastics)
    macd_hist = float(ind["macd_histogram"][idx]) if idx < ind["macd_histogram"].shape[0] else 0.0
    velocity = 0.0
    if idx >= 3 and atr_m15 > 0:
        velocity = float(closes[idx] - closes[idx - 3]) / atr_m15
    acceleration = 0.0
    if idx >= 6 and atr_m15 > 0:
        v1 = float(closes[idx - 3] - closes[idx - 6]) / atr_m15
        acceleration = velocity - v1

    spread = float((ohlcv.high[idx] - ohlcv.low[idx]) / pip) if pip > 0 else 0.0
    tick_volume = float(ohlcv.volume[idx]) if ohlcv.volume.size > idx else 0.0
    vol_z = _volume_zscore(ohlcv.volume, idx)
    liquidity = max(0.0, min(1.0, 0.5 + 0.25 * vol_z - 0.05 * spread))

    london_open = ts.normalize() + pd.Timedelta(hours=LONDON_OPEN_HOUR)
    ny_open = ts.normalize() + pd.Timedelta(hours=NY_OPEN_HOUR)
    minutes_from_london = float((ts - london_open).total_seconds() / 60.0)
    minutes_from_ny = float((ts - ny_open).total_seconds() / 60.0)

    news_min = int(minutes_to_news)
    minutes_to_major = float(news_min if news_min < 999 else 999)
    minutes_after_major = float(max(0, -news_min)) if news_min < 0 else 0.0

    llm_decision = _resolve_llm_decision(decision_source, llm_confidence)

    row: dict[str, Any] = {
        "trade_id": trade_id,
        "run_id": run_id,
        "wft_window": wft_window,
        "is_oos": is_oos,
        "symbol": setup.pair,
        "direction": setup.direction,
        "entry_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "session": session,
        "weekday": int(ts.weekday()),
        "month": int(ts.month),
        "is_gotobi": int(_is_gotobi(ts.date())),
        "hour": int(ts.hour),
        "minute": int(ts.minute),
        "setup_type": setup.setup_type,
        "fib_level": round(float(setup.retracement), 4),
        "fib_distance": round(fib_distance, 4),
        "pullback_depth": round(float(setup.retracement), 4),
        "swing_size": round(swing_size, 4),
        "swing_duration": swing_duration,
        "trend_direction": setup.direction,
        "trend_strength": round(abs(h1_macd_spread), 6),
        "ema20": round(ema20, 5),
        "ema50": round(ema50, 5),
        "ema200": round(ema200, 5),
        "ema20_slope": round(_ema_slope(closes, 20), 6),
        "ema50_slope": round(_ema_slope(closes, 50), 6),
        "ema200_slope": round(_ema_slope(closes, 200), 6),
        "ema_alignment_score": align,
        "atr_m15": round(float(atr_m15), 6),
        "atr_h1": round(float(atr_h1), 6) if not np.isnan(atr_h1) else None,
        "atr_h4": round(float(atr_h4), 6),
        "volatility_regime": round(_volatility_regime(atr_m15_arr, idx), 4),
        "rsi_m15": round(_rsi_last(closes, 14), 2),
        "rsi_h1": round(float(rsi_h1), 2) if not np.isnan(rsi_h1) else None,
        "momentum_score": round(float(score), 2),
        "velocity": round(velocity, 4),
        "acceleration": round(acceleration, 4),
        "distance_to_asia_high": round(_dist_pips(asia_hi), 2),
        "distance_to_asia_low": round(_dist_pips(asia_lo), 2),
        "distance_to_london_high": round(_dist_pips(london_hi), 2),
        "distance_to_london_low": round(_dist_pips(london_lo), 2),
        "distance_to_daily_high": round(_dist_pips(day_hi), 2),
        "distance_to_daily_low": round(_dist_pips(day_lo), 2),
        "spread": round(spread, 2),
        "tick_volume": tick_volume,
        "volume_zscore": round(vol_z, 4),
        "liquidity_score": round(liquidity, 4),
        "minutes_from_london_open": round(minutes_from_london, 1),
        "minutes_from_ny_open": round(minutes_from_ny, 1),
        "minutes_to_major_news": minutes_to_major,
        "minutes_after_major_news": minutes_after_major,
        "llm_decision": llm_decision,
        "llm_confidence": int(llm_confidence),
        "llm_reason": (llm_reason or "")[:500],
        "decision_source": decision_source,
        "executed": 0,
        "ev_rank": None,
        "ev_bucket": None,
    }
    out = {k: row.get(k) for k in DN_FEATURE_COLUMNS if k in row}
    if is_dn_prop_gate_enabled():
        from src.filters.dn_bayes_ev_v2 import apply_dn_ev_v2_to_row

        apply_dn_ev_v2_to_row(out)
        from src.filters.dn_prop_gate_v1 import evaluate_dn_prop_gate_sizing

        gate = evaluate_dn_prop_gate_sizing(out)
        out["ev_rank"] = gate["ev_rank_v2"]
        out["ev_bucket"] = gate["tier"]
    elif is_dn_ev_rank_enabled():
        apply_dn_ev_to_row(out)
    return out


def build_dn_exit_patch(
    *,
    setup: DiNapoliSetup,
    trade_result: str,
    profit_r: float,
    shadow_result: str,
    shadow_profit_r: float,
    holding_minutes: int,
    peak_unrealized_r: float,
    pair_df: Any,
    start_index: int,
    max_holding_bars: int,
    executed: bool,
) -> dict[str, Any]:
    mfe, mae, _ = compute_excursions_r(
        pair_df,
        start_index,
        setup.direction,
        setup.entry_price,
        setup.stop_loss,
        max_holding_bars=max_holding_bars,
    )
    if peak_unrealized_r > 0:
        mfe = max(mfe, float(peak_unrealized_r))

    if executed and trade_result in ("WIN", "LOSS"):
        result_r = float(profit_r)
        win_loss = trade_result
    else:
        result_r = float(shadow_profit_r)
        win_loss = shadow_result if shadow_result in ("WIN", "LOSS") else "NONE"

    exit_ts = pd.Timestamp(setup.timestamp) + pd.Timedelta(minutes=int(holding_minutes or 0))

    return {
        "exit_time": exit_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "result_r": round(result_r, 4),
        "win_loss": win_loss,
        "holding_minutes": int(holding_minutes or 0),
        "max_favorable_excursion_r": round(mfe, 4),
        "max_adverse_excursion_r": round(mae, 4),
        "executed": int(executed),
    }
