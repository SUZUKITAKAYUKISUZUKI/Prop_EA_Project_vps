"""DiNapoli MTF helpers — exec M15 / structure H1 / ATR H4 (numpy boundary)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from strategies.bt_ohlcv import BtOhlcvFrame, as_ohlcv, ts_ns_to_pd
from strategies.archive.cspa_arrays import compute_atr_np
from strategies.dinapoli_universe_fast import (
    SIGNAL_BUY,
    SIGNAL_SELL,
    DiNapoliUniverseFast,
)

ATR_PERIOD = 14


def map_htf_index(exec_datetime_ns: np.ndarray, htf_datetime_ns: np.ndarray) -> np.ndarray:
    """For each exec bar, index of the last HTF bar with timestamp <= exec ts."""
    n = exec_datetime_ns.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    if htf_datetime_ns.size == 0:
        return out
    for i in range(n):
        idx = int(np.searchsorted(htf_datetime_ns, exec_datetime_ns[i], side="right") - 1)
        if idx >= 0:
            out[i] = idx
    return out


def compute_h4_atr_on_exec(exec_dt_ns: np.ndarray, h4: Any, period: int = ATR_PERIOD) -> np.ndarray:
    """Map H4 ATR(period) onto each exec bar (numpy arrays only)."""
    h4_arr = as_ohlcv(h4)
    atr_h4 = compute_atr_np(h4_arr.high, h4_arr.low, h4_arr.close, period)
    mapped = np.full(exec_dt_ns.shape[0], np.nan, dtype=np.float64)
    htf_idx = map_htf_index(exec_dt_ns, h4_arr.datetime_ns)
    for i in range(exec_dt_ns.shape[0]):
        j = htf_idx[i]
        if j >= 0 and not np.isnan(atr_h4[j]):
            mapped[i] = atr_h4[j]
    return mapped


def h1_structure_confirms(
    direction: int,
    exec_idx: int,
    h1_idx: int,
    h1_macd_line: np.ndarray,
    h1_macd_signal: np.ndarray,
) -> bool:
    if h1_idx < 0 or h1_idx >= h1_macd_line.shape[0]:
        return False
    if np.isnan(h1_macd_line[h1_idx]) or np.isnan(h1_macd_signal[h1_idx]):
        return False
    if direction > 0:
        return h1_macd_line[h1_idx] > h1_macd_signal[h1_idx]
    return h1_macd_line[h1_idx] < h1_macd_signal[h1_idx]


def detect_dinapoli_setups_for_pair(
    df: pd.DataFrame | BtOhlcvFrame,
    pair: str,
    h1_df: pd.DataFrame | BtOhlcvFrame | None = None,
    h4_df: pd.DataFrame | BtOhlcvFrame | None = None,
    *,
    engine: DiNapoliUniverseFast | None = None,
    progress_hook: Callable[[int], None] | None = None,
    resume_from_bar: int | None = None,
    initial_setups: list[Any] | None = None,
    on_checkpoint: Callable[[int, list[Any], dict[str, Any] | None], None] | None = None,
    checkpoint_every: int = 0,
    **kwargs: Any,
) -> list[Any]:
    del kwargs
    from strategies.dinapoli import (
        ALLOWED_PAIRS,
        DiNapoliSetup,
        _build_setup_from_signal,
    )

    if pair not in ALLOWED_PAIRS:
        return []

    exec_arr = as_ohlcv(df)
    if exec_arr.length < 60:
        return []

    engine = engine or DiNapoliUniverseFast()
    indicators, structures, signals = engine.run_pipeline(
        exec_arr.high,
        exec_arr.low,
        exec_arr.close,
        exec_arr.datetime_ns,
    )

    h1_macd_line: np.ndarray | None = None
    h1_macd_signal: np.ndarray | None = None
    h1_dt_ns: np.ndarray | None = None
    if h1_df is not None:
        h1_arr = as_ohlcv(h1_df)
        h1_ind = engine.compute_all_indicators(h1_arr.high, h1_arr.low, h1_arr.close)
        h1_macd_line = h1_ind["macd_line"]
        h1_macd_signal = h1_ind["macd_signal"]
        h1_dt_ns = h1_arr.datetime_ns

    h4_atr_on_exec = (
        compute_h4_atr_on_exec(exec_arr.datetime_ns, h4_df)
        if h4_df is not None
        else np.full(exec_arr.length, np.nan, dtype=np.float64)
    )

    h1_map = map_htf_index(exec_arr.datetime_ns, h1_dt_ns) if h1_dt_ns is not None else None
    setups: list[DiNapoliSetup] = list(initial_setups) if initial_setups else []
    loop_start = max(0, int(resume_from_bar)) if resume_from_bar is not None else 0
    step = max(exec_arr.length // 100, 1)
    next_ckpt = loop_start + checkpoint_every if checkpoint_every > 0 else exec_arr.length + 1

    for i in range(loop_start, exec_arr.length):
        if progress_hook is not None and i % step == 0:
            progress_hook(i)
        sig = int(signals[i])
        if sig == SIGNAL_BUY:
            direction = "BUY"
            dir_int = 1
        elif sig == SIGNAL_SELL:
            direction = "SELL"
            dir_int = -1
        else:
            if on_checkpoint is not None and i >= next_ckpt:
                on_checkpoint(i + 1, setups, None)
                next_ckpt = i + 1 + checkpoint_every
            continue

        if h1_map is not None and h1_macd_line is not None and h1_macd_signal is not None:
            if not h1_structure_confirms(dir_int, i, int(h1_map[i]), h1_macd_line, h1_macd_signal):
                if on_checkpoint is not None and i >= next_ckpt:
                    on_checkpoint(i + 1, setups, None)
                    next_ckpt = i + 1 + checkpoint_every
                continue

        atr_proxy = float(h4_atr_on_exec[i])
        if np.isnan(atr_proxy) or atr_proxy <= 0.0:
            atr_proxy = float(exec_arr.high[i] - exec_arr.low[i])

        setups.append(
            _build_setup_from_signal(
                bar_index=i,
                timestamp=ts_ns_to_pd(int(exec_arr.datetime_ns[i])),
                pair=pair,
                direction=direction,
                signal=sig,
                close=float(exec_arr.close[i]),
                atr_proxy=atr_proxy,
                indicators=indicators,
                structures=structures,
            )
        )

        if on_checkpoint is not None and i >= next_ckpt:
            on_checkpoint(i + 1, setups, None)
            next_ckpt = i + 1 + checkpoint_every

    if progress_hook is not None:
        progress_hook(exec_arr.length)
    if on_checkpoint is not None:
        on_checkpoint(exec_arr.length, setups, None)
    return setups
