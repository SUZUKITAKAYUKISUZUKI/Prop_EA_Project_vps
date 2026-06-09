"""
strategies/bt_ohlcv.py — Backtest OHLCV layer (numpy-only hot paths).

CSV load / resample / merge / clip produce ``BtOhlcvFrame`` (wraps ``OhlcvArrays``).
Pandas is not used inside backtest scan, simulation, or L5 tracking loops.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from strategies.cspa_arrays import (
    OhlcvArrays,
    atr_at_index,
    compute_atr_np,
    datetime_ns_from_column,
    timestamp_from_ns,
)

# 3y window (aligned with backtest_runner.H1_3Y_WINDOW_*)
BT_WINDOW_START_NS = int(np.datetime64("2023-01-02T00:00:00", "ns").astype(np.int64))
BT_WINDOW_END_NS = int(np.datetime64("2026-04-30T23:59:59", "ns").astype(np.int64))

LIVE_BAR_BUFFER_MAX = 500

_OHLCV_REGISTRY: dict[int, OhlcvArrays] = {}


def register_ohlcv(obj: Any, arrays: OhlcvArrays) -> None:
    _OHLCV_REGISTRY[id(obj)] = arrays


def lookup_ohlcv(obj: Any) -> OhlcvArrays | None:
    if isinstance(obj, BtOhlcvFrame):
        return obj.arrays
    if isinstance(obj, OhlcvArrays):
        return obj
    return _OHLCV_REGISTRY.get(id(obj))


def as_ohlcv(obj: Any) -> OhlcvArrays:
    found = lookup_ohlcv(obj)
    if found is not None:
        return found
    if isinstance(obj, pd.DataFrame):
        arr = OhlcvArrays.from_prepared_df(obj)
        register_ohlcv(obj, arr)
        return arr
    raise TypeError(f"expected BtOhlcvFrame, OhlcvArrays, or DataFrame, got {type(obj)!r}")


def ts_ns_to_pd(ts_ns: int) -> pd.Timestamp:
    """Naive bar timestamp from OHLCV ``datetime_ns`` (timezone-independent)."""
    return pd.Timestamp(int(ts_ns))


def normalize_ts_ns(ts: pd.Timestamp | str | int | np.datetime64) -> int:
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    if isinstance(ts, np.datetime64):
        return int(ts.astype("datetime64[ns]").astype(np.int64))
    return int(pd.Timestamp(ts).value)


def load_ohlcv_csv(path: Path) -> OhlcvArrays:
    """Load FT6 OHLCV CSV without pandas."""
    dt_list: list[int] = []
    o: list[float] = []
    h: list[float] = []
    l: list[float] = []
    c: list[float] = []
    v: list[float] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = {name.strip("<>").upper(): idx for idx, name in enumerate(header)}
        for row in reader:
            if not row:
                continue
            ymd = row[cols["DTYYYYMMDD"]].strip()
            t = row[cols["TIME"]].strip().zfill(4)
            dt = datetime.strptime(ymd + t, "%Y%m%d%H%M")
            dt_list.append(int(np.datetime64(dt, "ns").astype(np.int64)))
            o.append(float(row[cols["OPEN"]]))
            h.append(float(row[cols["HIGH"]]))
            l.append(float(row[cols["LOW"]]))
            c.append(float(row[cols["CLOSE"]]))
            vol_key = "VOL" if "VOL" in cols else "VOLUME"
            v.append(float(row[cols[vol_key]]) if vol_key in cols else 0.0)
    if not dt_list:
        return OhlcvArrays(
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            volume=np.array([], dtype=np.float64),
            datetime_ns=np.array([], dtype=np.int64),
        )
    order = np.argsort(dt_list)
    return OhlcvArrays(
        open=np.asarray(o, dtype=np.float64)[order],
        high=np.asarray(h, dtype=np.float64)[order],
        low=np.asarray(l, dtype=np.float64)[order],
        close=np.asarray(c, dtype=np.float64)[order],
        volume=np.asarray(v, dtype=np.float64)[order],
        datetime_ns=np.asarray(dt_list, dtype=np.int64)[order],
    )


@dataclass
class BtOhlcvFrame:
    """Backtest OHLCV container — hot paths use ``.arrays`` only."""

    arrays: OhlcvArrays

    def __post_init__(self) -> None:
        register_ohlcv(self, self.arrays)

    @classmethod
    def from_csv(cls, path: Path) -> BtOhlcvFrame:
        return cls(load_ohlcv_csv(path))

    @classmethod
    def from_arrays(cls, arrays: OhlcvArrays) -> BtOhlcvFrame:
        return cls(arrays)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> BtOhlcvFrame:
        work = df.sort_values("datetime").reset_index(drop=True)
        return cls(OhlcvArrays.from_prepared_df(work))

    @classmethod
    def make_empty(cls) -> BtOhlcvFrame:
        return cls(
            OhlcvArrays(
                open=np.array([], dtype=np.float64),
                high=np.array([], dtype=np.float64),
                low=np.array([], dtype=np.float64),
                close=np.array([], dtype=np.float64),
                volume=np.array([], dtype=np.float64),
                datetime_ns=np.array([], dtype=np.int64),
            )
        )

    def __len__(self) -> int:
        return self.arrays.length

    @property
    def empty(self) -> bool:
        return self.arrays.length == 0

    def clip_window(
        self,
        start_ns: int = BT_WINDOW_START_NS,
        end_ns: int = BT_WINDOW_END_NS,
    ) -> BtOhlcvFrame:
        dt = self.arrays.datetime_ns
        mask = (dt >= start_ns) & (dt <= end_ns)
        if not mask.any():
            return BtOhlcvFrame.make_empty()
        return BtOhlcvFrame(take_arrays(self.arrays, mask))

    def slice_end(self, end_index: int) -> BtOhlcvFrame:
        end_index = min(max(end_index, -1), self.arrays.length - 1)
        if end_index < 0:
            return BtOhlcvFrame.make_empty()
        sl = slice(0, end_index + 1)
        a = self.arrays
        return BtOhlcvFrame(
            OhlcvArrays(
                open=a.open[sl],
                high=a.high[sl],
                low=a.low[sl],
                close=a.close[sl],
                volume=a.volume[sl],
                datetime_ns=a.datetime_ns[sl],
            )
        )

    def datetime_at(self, index: int) -> pd.Timestamp:
        return ts_ns_to_pd(int(self.arrays.datetime_ns[index]))

    def to_pandas(self) -> pd.DataFrame:
        """Cold-path export only (reports/tests)."""
        a = self.arrays
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(a.datetime_ns),
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
            }
        )


def _resample_ohlcv(arr: OhlcvArrays, bar_ns: int) -> OhlcvArrays:
    if arr.length == 0:
        return arr
    bucket = (arr.datetime_ns // bar_ns) * bar_ns
    uniq, inv = np.unique(bucket, return_inverse=True)
    n = len(uniq)
    o = np.zeros(n, dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)
    l = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=bool)
    for i in range(arr.length):
        b = int(inv[i])
        if not seen[b]:
            o[b] = arr.open[i]
            h[b] = arr.high[i]
            l[b] = arr.low[i]
            seen[b] = True
        else:
            h[b] = max(h[b], arr.high[i])
            l[b] = min(l[b], arr.low[i])
        c[b] = arr.close[i]
        v[b] += arr.volume[i]
    valid = seen
    return OhlcvArrays(
        open=o[valid],
        high=h[valid],
        low=l[valid],
        close=c[valid],
        volume=v[valid],
        datetime_ns=uniq[valid],
    )


def resample_to_h1(frame: BtOhlcvFrame) -> BtOhlcvFrame:
    bar_ns = int(np.timedelta64(1, "h") / np.timedelta64(1, "ns"))
    return BtOhlcvFrame.from_arrays(_resample_ohlcv(frame.arrays, bar_ns))


def resample_to_m15(frame: BtOhlcvFrame) -> BtOhlcvFrame:
    bar_ns = int(np.timedelta64(15, "m") / np.timedelta64(1, "ns"))
    return BtOhlcvFrame.from_arrays(_resample_ohlcv(frame.arrays, bar_ns))


def resample_bars_ns(frame: BtOhlcvFrame, bar_ns: int) -> BtOhlcvFrame:
    if frame.empty:
        return BtOhlcvFrame.make_empty()
    return BtOhlcvFrame.from_arrays(_resample_ohlcv(frame.arrays, bar_ns))


@dataclass(frozen=True, slots=True)
class MergedBars:
    datetime_ns: np.ndarray
    open_gbp: np.ndarray
    high_gbp: np.ndarray
    low_gbp: np.ndarray
    close_gbp: np.ndarray
    volume_gbp: np.ndarray
    open_eur: np.ndarray
    high_eur: np.ndarray
    low_eur: np.ndarray
    close_eur: np.ndarray
    volume_eur: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.datetime_ns))

    def __len__(self) -> int:
        return self.length

    @classmethod
    def merge(cls, gbp: BtOhlcvFrame, eur: BtOhlcvFrame) -> MergedBars:
        g, e = gbp.arrays, eur.arrays
        idx_g = {int(ns): i for i, ns in enumerate(g.datetime_ns)}
        rows: list[tuple[int, int, int]] = []
        for j, ns in enumerate(e.datetime_ns):
            i = idx_g.get(int(ns))
            if i is not None:
                rows.append((int(ns), i, j))
        rows.sort(key=lambda x: x[0])
        if not rows:
            empty = np.array([], dtype=np.float64)
            empty_ns = np.array([], dtype=np.int64)
            return cls(empty_ns, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty)
        ns = np.array([r[0] for r in rows], dtype=np.int64)
        gi = np.array([r[1] for r in rows], dtype=np.int64)
        ei = np.array([r[2] for r in rows], dtype=np.int64)
        return cls(
            ns,
            g.open[gi],
            g.high[gi],
            g.low[gi],
            g.close[gi],
            g.volume[gi],
            e.open[ei],
            e.high[ei],
            e.low[ei],
            e.close[ei],
            e.volume[ei],
        )

    def timestamp_at(self, index: int) -> pd.Timestamp:
        return ts_ns_to_pd(int(self.datetime_ns[index]))


def merge_bars(gbp: BtOhlcvFrame, eur: BtOhlcvFrame) -> MergedBars:
    return MergedBars.merge(gbp, eur)


def load_pair_data(gbp_path: Path, eur_path: Path) -> tuple[BtOhlcvFrame, BtOhlcvFrame]:
    return BtOhlcvFrame.from_csv(gbp_path), BtOhlcvFrame.from_csv(eur_path)


def clip_ohlcv_window(
    frame: BtOhlcvFrame,
    start_ns: int = BT_WINDOW_START_NS,
    end_ns: int = BT_WINDOW_END_NS,
) -> BtOhlcvFrame:
    return frame.clip_window(start_ns, end_ns)


def asof_end_index(arr: OhlcvArrays, ts_ns: int) -> int:
    return int(np.searchsorted(arr.datetime_ns, ts_ns, side="right") - 1)


def resolve_bar_position_np(arr: OhlcvArrays, ts_ns: int) -> int | None:
    pos = np.searchsorted(arr.datetime_ns, ts_ns, side="left")
    if pos < arr.length and int(arr.datetime_ns[pos]) == int(ts_ns):
        return int(pos)
    if pos > 0 and int(arr.datetime_ns[pos - 1]) == int(ts_ns):
        return int(pos - 1)
    return None


def resolve_track_start_index_np(arr: OhlcvArrays, ts_ns: int) -> int:
    pos = resolve_bar_position_np(arr, ts_ns)
    if pos is not None:
        return pos
    padded = int(np.searchsorted(arr.datetime_ns, ts_ns, side="right") - 1)
    return max(0, padded)


def find_first_bar_at_or_after_np(arr: OhlcvArrays, start_index: int, deadline_ns: int) -> int | None:
    for i in range(start_index + 1, arr.length):
        if int(arr.datetime_ns[i]) >= int(deadline_ns):
            return i
    return None


def bars_to_dict_list(arr: OhlcvArrays) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(arr.length):
        out.append(
            {
                "time": timestamp_from_ns(int(arr.datetime_ns[i])).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(arr.open[i]),
                "high": float(arr.high[i]),
                "low": float(arr.low[i]),
                "close": float(arr.close[i]),
                "volume": float(arr.volume[i]),
            }
        )
    return out


def upsert_live_bars(existing: BtOhlcvFrame, incoming: BtOhlcvFrame) -> BtOhlcvFrame:
    if existing.empty:
        return incoming
    if incoming.empty:
        return existing
    combined = OhlcvArrays(
        open=np.concatenate([existing.arrays.open, incoming.arrays.open]),
        high=np.concatenate([existing.arrays.high, incoming.arrays.high]),
        low=np.concatenate([existing.arrays.low, incoming.arrays.low]),
        close=np.concatenate([existing.arrays.close, incoming.arrays.close]),
        volume=np.concatenate([existing.arrays.volume, incoming.arrays.volume]),
        datetime_ns=np.concatenate([existing.arrays.datetime_ns, incoming.arrays.datetime_ns]),
    )
    order = np.argsort(combined.datetime_ns, kind="mergesort")
    dt = combined.datetime_ns[order]
    keep = np.concatenate([[True], dt[1:] != dt[:-1]])
    idx = order[keep]
    trimmed = OhlcvArrays(
        open=combined.open[idx],
        high=combined.high[idx],
        low=combined.low[idx],
        close=combined.close[idx],
        volume=combined.volume[idx],
        datetime_ns=combined.datetime_ns[idx],
    )
    if trimmed.length > LIVE_BAR_BUFFER_MAX:
        sl = slice(trimmed.length - LIVE_BAR_BUFFER_MAX, trimmed.length)
        trimmed = OhlcvArrays(
            open=trimmed.open[sl],
            high=trimmed.high[sl],
            low=trimmed.low[sl],
            close=trimmed.close[sl],
            volume=trimmed.volume[sl],
            datetime_ns=trimmed.datetime_ns[sl],
        )
    return BtOhlcvFrame(trimmed)


def bars_payload_to_frame(bars: list[dict[str, Any]]) -> BtOhlcvFrame:
    if not bars:
        return BtOhlcvFrame.make_empty()
    dt: list[int] = []
    o, h, l, c, v = [], [], [], [], []
    for bar in bars:
        ts = pd.Timestamp(bar["time"])
        dt.append(normalize_ts_ns(ts))
        o.append(float(bar["open"]))
        h.append(float(bar["high"]))
        l.append(float(bar["low"]))
        c.append(float(bar["close"]))
        v.append(float(bar.get("volume", 0.0)))
    return BtOhlcvFrame(
        OhlcvArrays(
            open=np.asarray(o, dtype=np.float64),
            high=np.asarray(h, dtype=np.float64),
            low=np.asarray(l, dtype=np.float64),
            close=np.asarray(c, dtype=np.float64),
            volume=np.asarray(v, dtype=np.float64),
            datetime_ns=np.asarray(dt, dtype=np.int64),
        )
    )


@dataclass(frozen=True, slots=True)
class MergedBarView:
    datetime_ns: int
    open_gbp: float
    high_gbp: float
    low_gbp: float
    close_gbp: float
    volume_gbp: float
    open_eur: float
    high_eur: float
    low_eur: float
    close_eur: float
    volume_eur: float


def merged_bar_at(merged: MergedBars, index: int) -> MergedBarView:
    i = int(index)
    return MergedBarView(
        datetime_ns=int(merged.datetime_ns[i]),
        open_gbp=float(merged.open_gbp[i]),
        high_gbp=float(merged.high_gbp[i]),
        low_gbp=float(merged.low_gbp[i]),
        close_gbp=float(merged.close_gbp[i]),
        volume_gbp=float(merged.volume_gbp[i]),
        open_eur=float(merged.open_eur[i]),
        high_eur=float(merged.high_eur[i]),
        low_eur=float(merged.low_eur[i]),
        close_eur=float(merged.close_eur[i]),
        volume_eur=float(merged.volume_eur[i]),
    )


def slice_merged(merged: MergedBars, *, max_bars: int | None = None) -> MergedBars:
    if max_bars is None or max_bars >= merged.length:
        return merged
    sl = slice(0, max(0, max_bars))
    return MergedBars(
        merged.datetime_ns[sl],
        merged.open_gbp[sl],
        merged.high_gbp[sl],
        merged.low_gbp[sl],
        merged.close_gbp[sl],
        merged.volume_gbp[sl],
        merged.open_eur[sl],
        merged.high_eur[sl],
        merged.low_eur[sl],
        merged.close_eur[sl],
        merged.volume_eur[sl],
    )


def unique_calendar_days(arr: OhlcvArrays) -> int:
    if arr.length == 0:
        return 0
    day_ns = int(np.timedelta64(1, "D") / np.timedelta64(1, "ns"))
    days = arr.datetime_ns // day_ns
    return int(len(np.unique(days)))


def resolve_event_loop_indices_np(merged: MergedBars, setup_ts_ns: set[int]) -> np.ndarray:
    if merged.length == 0 or not setup_ts_ns:
        return np.array([], dtype=np.int64)
    mask = np.isin(merged.datetime_ns, list(setup_ts_ns))
    return np.flatnonzero(mask)


def ohlcv_fingerprint(arr: OhlcvArrays) -> str:
    if arr.length == 0:
        return "0::"
    return (
        f"{arr.length}:"
        f"{timestamp_from_ns(int(arr.datetime_ns[0]))}:"
        f"{timestamp_from_ns(int(arr.datetime_ns[-1]))}"
    )


def write_records_csv(path: Path, columns: tuple[str, ...] | list[str], records: list[dict[str, Any]]) -> None:
    """Write BT result CSV without pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in columns})


# Extend OhlcvArrays with slice helpers used by scan hot paths
def take_arrays(arr: OhlcvArrays, mask: np.ndarray) -> OhlcvArrays:
    return OhlcvArrays(
        open=arr.open[mask],
        high=arr.high[mask],
        low=arr.low[mask],
        close=arr.close[mask],
        volume=arr.volume[mask],
        datetime_ns=arr.datetime_ns[mask],
    )


def slice_arrays(arr: OhlcvArrays, start: int, end: int) -> OhlcvArrays:
    sl = slice(max(0, start), min(end + 1, arr.length))
    return OhlcvArrays(
        open=arr.open[sl],
        high=arr.high[sl],
        low=arr.low[sl],
        close=arr.close[sl],
        volume=arr.volume[sl],
        datetime_ns=arr.datetime_ns[sl],
    )
