"""Timestamp-based bar lookup and MTF alignment (numpy-first; pandas legacy fallback)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

MergeHow = Literal["left", "inner", "right", "outer"]


def normalize_bar_timestamp(timestamp: pd.Timestamp | str) -> pd.Timestamp:
    """Normalize to tz-naive Timestamp for OHLCV datetime column comparison."""
    return pd.Timestamp(timestamp)


def resolve_bar_position(
    df: Any,
    timestamp: pd.Timestamp | str,
    *,
    datetime_col: str = "datetime",
) -> int | None:
    """Positional index of the bar at ``timestamp`` within ``df`` (same-frame only)."""
    from strategies.bt_ohlcv import as_ohlcv, normalize_ts_ns, resolve_bar_position_np

    try:
        return resolve_bar_position_np(as_ohlcv(df), normalize_ts_ns(timestamp))
    except TypeError:
        pass

    if datetime_col not in df.columns:
        return None
    ts = normalize_bar_timestamp(timestamp)
    matches = df.loc[df[datetime_col] == ts]
    if matches.empty:
        return None
    from strategies.market_utils import positional_index

    return positional_index(df, matches.index[0])


def resolve_track_start_index(
    df: Any,
    entry_timestamp: pd.Timestamp | str,
    *,
    datetime_col: str = "datetime",
) -> int:
    """
    Timestamp-synchronized L5 start index (no cross-frame iloc reuse).

    Uses exact bar match when available, otherwise pad (last bar at or before entry).
    """
    from strategies.bt_ohlcv import as_ohlcv, normalize_ts_ns, resolve_track_start_index_np

    try:
        return resolve_track_start_index_np(as_ohlcv(df), normalize_ts_ns(entry_timestamp))
    except TypeError:
        pass

    ts = normalize_bar_timestamp(entry_timestamp)
    pos = resolve_bar_position(df, ts, datetime_col=datetime_col)
    if pos is not None:
        return pos

    if datetime_col not in df.columns:
        raise KeyError(f"datetime column {datetime_col!r} missing from frame")

    dt_values = pd.to_datetime(df[datetime_col], errors="coerce")
    dt_index = pd.DatetimeIndex(dt_values)
    padded = dt_index.get_indexer([ts], method="pad")
    start_index = int(padded[0])
    if start_index < 0:
        return 0
    return start_index


def loc_bar_at_timestamp(
    df: pd.DataFrame,
    timestamp: pd.Timestamp | str,
    *,
    datetime_col: str = "datetime",
) -> pd.Series | None:
    """Return the single bar row whose ``datetime_col`` equals ``timestamp``, or None."""
    if datetime_col not in df.columns:
        return None
    ts = normalize_bar_timestamp(timestamp)
    matches = df.loc[df[datetime_col] == ts]
    if matches.empty:
        return None
    return matches.iloc[0]


def merge_bars_by_timestamp(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str = "datetime",
    how: MergeHow = "left",
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    """Align two OHLCV frames on bar open time — never on positional index."""
    if on not in left.columns or on not in right.columns:
        raise KeyError(f"merge key {on!r} must exist in both frames")
    return left.merge(right, on=on, how=how, suffixes=suffixes)


def assert_same_frame_timestamp(
    df: pd.DataFrame,
    timestamp: pd.Timestamp | str,
    *,
    bar_index: int | None = None,
    datetime_col: str = "datetime",
) -> None:
    """Raise if ``bar_index`` does not point to ``timestamp`` on ``df``."""
    if bar_index is None:
        return
    row = loc_bar_at_timestamp(df, timestamp, datetime_col=datetime_col)
    if row is None:
        raise LookupError(f"timestamp {timestamp!r} not found in frame")
    from strategies.market_utils import positional_index

    pos = positional_index(df, row.name)
    if pos != bar_index:
        raise ValueError(
            f"bar_index {bar_index} does not match timestamp {timestamp!r} (expected pos {pos})"
        )
