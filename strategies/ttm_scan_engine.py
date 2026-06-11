"""TTM scan engine — numpy/numba detect path (LOW_UPDATE x SHORT flow features)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from strategies.bt_ohlcv import BtOhlcvFrame, as_ohlcv
from strategies.ttm_arrays import TtmScanArrays, build_ttm_scan_arrays
from strategies.ttm_scan_numba import scan_ttm_events_numba, ttm_scan_numba_active

logger = logging.getLogger(__name__)

PATTERN_BY_CODE: dict[int, str] = {5: "TTM_LOW_UPDATE"}
JST = timezone(timedelta(hours=9))


def _frame_arrays(obj: Any) -> BtOhlcvFrame:
    if isinstance(obj, BtOhlcvFrame):
        return obj
    return BtOhlcvFrame.from_arrays(as_ohlcv(obj))


def _is_month_end_dt(dt_ns: int) -> bool:
    jst = datetime.fromtimestamp(int(dt_ns) / 1_000_000_000, tz=timezone.utc).astimezone(JST)
    nxt = jst + timedelta(days=1)
    return nxt.month != jst.month


def _is_quarter_end_dt(dt_ns: int) -> bool:
    jst = datetime.fromtimestamp(int(dt_ns) / 1_000_000_000, tz=timezone.utc).astimezone(JST)
    return _is_month_end_dt(dt_ns) and jst.month in (3, 6, 9, 12)


def scan_ttm_setups_from_frames(
    m1: Any,
    m5: Any,
    m15: Any,
    pair: str,
    *,
    pip: float,
    max_events_per_day: int,
    bar_minutes: int,
    progress_hook: Any | None = None,
    min_bar_index: int | None = None,
) -> list[Any]:
    from strategies.ttm import (
        ALLOWED_PAIRS,
        TTM_SHORT_EVENT_TYPE,
        TTM_SHORT_TRIGGER,
        TTM_TRUST_SCORE,
        TtmFeatures,
        TtmSetup,
        _minute_bucket,
        _session_label,
        _week_of_month,
        derive_ttm_event_type,
    )

    if pair not in ALLOWED_PAIRS:
        return []

    m1_f = _frame_arrays(m1)
    m5_f = _frame_arrays(m5)
    if m15 is None:
        raise ValueError("TTM scan requires M15 frame for ATR")
    m15_f = _frame_arrays(m15)
    if m1_f.empty or m1_f.arrays.length < 20:
        return []

    ctx = build_ttm_scan_arrays(m1_f, m5_f, m15_f)
    event_indices = ctx.event_indices
    if min_bar_index is not None and min_bar_index > 0:
        event_indices = event_indices[event_indices >= min_bar_index]
    if event_indices.size == 0:
        return []

    if not ttm_scan_numba_active():
        raise RuntimeError("TTM scan requires numba (set TTM_SCAN_NUMBA=1 and install numba)")

    raw = scan_ttm_events_numba(
        ctx.m1.open,
        ctx.m1.high,
        ctx.m1.low,
        ctx.m1.close,
        ctx.m1.datetime_ns,
        ctx.m1_minute_jst,
        ctx.m1_day_jst,
        ctx.m5_atr_on_m1,
        ctx.m15_atr_on_m1,
        ctx.h1_atr_on_m1,
        ctx.asia_high_by_day,
        ctx.asia_low_by_day,
        event_indices,
        float(pip),
        int(max_events_per_day),
        int(bar_minutes),
    )
    count = int(raw[-1])
    if count == 0:
        logger.info("TTM scan %s: 0 events", pair)
        return []

    if progress_hook is not None:
        progress_hook(len(ctx.event_indices))

    (
        sim_idx,
        _full_idx,
        dt_ns,
        _direction,
        entry,
        sl,
        tp,
        trigger,
        pattern,
        mins_to_ttm,
        _weekday_numba,
        atr_m5,
        atr_m15,
        atr_h1,
        tokyo_range,
        dist_al,
        asia_pct,
        pre_ret,
        pre_vel,
        pre_atr,
        low_break_dist,
        low_break_vel,
    ) = raw[:-1]

    setups: list[TtmSetup] = []
    for k in range(count):
        jst = datetime.fromtimestamp(int(dt_ns[k]) / 1_000_000_000, tz=timezone.utc).astimezone(JST)
        mins = float(mins_to_ttm[k])
        if int(trigger[k]) != 2:
            continue
        trigger_str = TTM_SHORT_TRIGGER
        pattern_str = PATTERN_BY_CODE.get(int(pattern[k]), "TTM_LOW_UPDATE")
        atr_m15_pips = float(atr_m15[k])
        atr_m5_pips = float(atr_m5[k])
        atr_h1_pips = float(atr_h1[k])
        tokyo_range_val = float(tokyo_range[k])
        dist_al_val = float(dist_al[k])
        asia_pct_val = float(asia_pct[k])
        pre_vel_val = float(pre_vel[k])
        pre_ret_val = float(pre_ret[k])
        low_break_dist_val = float(low_break_dist[k])
        low_break_vel_val = float(low_break_vel[k])
        is_gotobi = jst.day in (5, 10, 15, 20, 25, 30)
        weekday = jst.weekday()
        event_type_str = derive_ttm_event_type(event_trigger=trigger_str)
        mins_after = max(0.0, float(jst.hour * 60 + jst.minute - (9 * 60 + 55)))
        atr_ratio_m5_h1 = atr_m5_pips / atr_h1_pips if atr_h1_pips > 1.0e-12 else 0.0
        atr_ratio_m15_h1 = atr_m15_pips / atr_h1_pips if atr_h1_pips > 1.0e-12 else 0.0

        features = TtmFeatures(
            pair=pair,
            timestamp=jst.strftime("%Y-%m-%d %H:%M:%S"),
            weekday=weekday,
            month=jst.month,
            quarter=(jst.month - 1) // 3 + 1,
            week_of_month=_week_of_month(jst.day),
            hour=jst.hour,
            minute=jst.minute,
            minute_bucket=_minute_bucket(jst.hour, jst.minute),
            session=_session_label(jst.hour),
            minutes_to_ttm=mins,
            minutes_after_ttm=mins_after,
            is_gotobi=is_gotobi,
            is_month_end=_is_month_end_dt(int(dt_ns[k])),
            is_quarter_end=_is_quarter_end_dt(int(dt_ns[k])),
            pre_ttm_return=pre_ret_val,
            pre_ttm_velocity=pre_vel_val,
            pre_ttm_range=tokyo_range_val,
            pre_ttm_atr_ratio=float(pre_atr[k]),
            asian_range=tokyo_range_val,
            asian_low_distance=dist_al_val,
            asian_range_pct=asia_pct_val,
            low_break_distance=low_break_dist_val,
            low_break_velocity=low_break_vel_val,
            atr_m5=atr_m5_pips,
            atr_m15=atr_m15_pips,
            atr_h1=atr_h1_pips,
            atr_ratio_m5_h1=atr_ratio_m5_h1,
            atr_ratio_m15_h1=atr_ratio_m15_h1,
            event_trigger=trigger_str,
            pattern_class=pattern_str,
            event_type=event_type_str,
        )

        setups.append(
            TtmSetup(
                timestamp=pd.Timestamp(int(dt_ns[k]), unit="ns"),
                pair=pair,
                direction="SHORT",
                entry_price=float(entry[k]),
                stop_loss=float(sl[k]),
                take_profit=float(tp[k]),
                candidate_score=TTM_TRUST_SCORE,
                bar_index=int(sim_idx[k]),
                ttm_features=features,
                event_trigger=trigger_str,  # type: ignore[arg-type]
                pattern_class=pattern_str,  # type: ignore[arg-type]
            )
        )

    logger.info("TTM Short scan %s: %d LOW_UPDATE events", pair, len(setups))
    return setups
