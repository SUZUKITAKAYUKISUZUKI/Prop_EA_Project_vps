"""
strategies/vamr_features.py — shared feature enrichment for VAMR Phase 1/2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.market_utils import pip_size_for_pair
from strategies.var_detector import VP_TOUCH_ATR_MULT
from strategies.var_reversal import ALLOWED_PAIRS

DATA_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"

PAIR_FILES = {
    "AUDNZD": ("audnzd_h1_10y.csv", "audnzd_h4_10y.csv"),
    "EURGBP": ("eurgbp_h1_10y.csv", "eurgbp_h4_10y.csv"),
    "USDCAD": ("usdcad_h1_10y.csv", "usdcad_h4_10y.csv"),
}


def profit_factor(values: pd.Series) -> float:
    r = pd.to_numeric(values, errors="coerce").dropna()
    if r.empty:
        return 0.0
    gw = r[r > 0].sum()
    gl = abs(r[r < 0].sum())
    if gl <= 0:
        return float("inf") if gw > 0 else 0.0
    return float(gw / gl)


def pf_str(pf: float) -> str:
    if np.isinf(pf):
        return "inf"
    if not np.isfinite(pf):
        return "n/a"
    return f"{pf:.3f}"


def session_type_utc(ts: pd.Timestamp) -> str:
    hour = int(pd.Timestamp(ts).hour)
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 22:
        return "NY"
    return "ASIA"


def primary_pa_type(pa_types: str) -> str:
    text = str(pa_types or "")
    for tag in ("PIN_BAR", "ENGULFING", "INSIDE_BAR", "CLOSE_ONLY"):
        if tag in text:
            return tag
    return "CLOSE_ONLY"


def _session_start_ts(ts: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(ts.date())
    start = day + pd.Timedelta(hours=7)
    if ts < start:
        start -= pd.Timedelta(days=1)
    return start


def load_pair_frames(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from main_platform import load_ohlcv

    h1_name, h4_name = PAIR_FILES[pair]
    h1 = load_ohlcv(DATA_DIR / h1_name)
    h4 = load_ohlcv(DATA_DIR / h4_name)
    for frame in (h1, h4):
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    return h1.reset_index(drop=True), h4.reset_index(drop=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["result_r"] = pd.to_numeric(out["result_r"], errors="coerce").fillna(0.0)
    out["atr_h1"] = pd.to_numeric(out["atr_h1"], errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["poc"] = pd.to_numeric(out["poc"], errors="coerce")
    out["vah"] = pd.to_numeric(out["vah"], errors="coerce")
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    out["value_area_width_atr_ratio"] = (out["vah"] - out["val"]) / out["atr_h1"].clip(lower=1e-9)
    out["price_vs_poc"] = (out["entry_price"] - out["poc"]) / out["atr_h1"].clip(lower=1e-9)
    out["abs_price_vs_poc"] = out["price_vs_poc"].abs()
    out["session_type"] = out["timestamp"].map(session_type_utc)
    out["primary_pa_type"] = out["pa_types"].map(primary_pa_type)
    return out


def enrich_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["retest_count"] = np.nan
    out["rejection_strength"] = np.nan
    out["volume_ratio_20ma"] = np.nan
    out["htf_aligned"] = pd.Series([None] * len(out), index=out.index, dtype="object")

    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for pair in sorted(out["pair"].dropna().unique()):
        pair_u = str(pair).upper()
        if pair_u not in PAIR_FILES:
            continue
        cache[pair_u] = load_pair_frames(pair_u)

    for pair, (h1, h4) in cache.items():
        mask = out["pair"].astype(str).str.upper() == pair
        sub = out.loc[mask]
        if sub.empty:
            continue
        h1_high = h1["high"].to_numpy(dtype=np.float64)
        h1_low = h1["low"].to_numpy(dtype=np.float64)
        h1_close = h1["close"].to_numpy(dtype=np.float64)
        h1_vol = h1["volume"].to_numpy(dtype=np.float64)
        h4_close = h4["close"].to_numpy(dtype=np.float64)
        h4_dt = pd.to_datetime(h4["datetime"]).to_numpy()
        h1_dt = pd.to_datetime(h1["datetime"]).to_numpy()

        for idx, row in sub.iterrows():
            bar_i = int(row["bar_index"])
            if bar_i < 0 or bar_i >= len(h1):
                continue
            atr = float(row["atr_h1"])
            vah = float(row["vah"])
            val = float(row["val"])
            direction = str(row["direction"])
            touch_buf = max(VP_TOUCH_ATR_MULT * atr, pip_size_for_pair(pair) * 2.0)

            sess_start = _session_start_ts(pd.Timestamp(row["timestamp"]))
            sess_start_i = int(np.searchsorted(h1_dt, np.datetime64(sess_start), side="left"))
            start_i = max(sess_start_i, 0)
            touches = 0
            for j in range(start_i, bar_i + 1):
                if direction == "SELL" and h1_high[j] >= vah - touch_buf:
                    touches += 1
                elif direction == "BUY" and h1_low[j] <= val + touch_buf:
                    touches += 1
            out.at[idx, "retest_count"] = touches

            hi = h1_high[bar_i]
            lo = h1_low[bar_i]
            cl = h1_close[bar_i]
            rng = hi - lo
            if rng > 0.0:
                if direction == "SELL":
                    out.at[idx, "rejection_strength"] = (hi - cl) / rng
                else:
                    out.at[idx, "rejection_strength"] = (cl - lo) / rng

            vol_start = max(0, bar_i - 19)
            base = float(np.mean(h1_vol[vol_start : bar_i + 1]))
            out.at[idx, "volume_ratio_20ma"] = float(h1_vol[bar_i] / base) if base > 0 else 1.0

            entry_ts = np.datetime64(pd.Timestamp(row["timestamp"]))
            h4_i = int(np.searchsorted(h4_dt, entry_ts, side="right")) - 1
            if h4_i >= 20:
                ma20 = float(np.mean(h4_close[h4_i - 19 : h4_i + 1]))
                htf_bull = h4_close[h4_i] > ma20
                if direction == "BUY":
                    out.at[idx, "htf_aligned"] = "aligned" if htf_bull else "not_aligned"
                else:
                    out.at[idx, "htf_aligned"] = "aligned" if not htf_bull else "not_aligned"

    out["retest_count"] = pd.to_numeric(out["retest_count"], errors="coerce")
    out["rejection_strength"] = pd.to_numeric(out["rejection_strength"], errors="coerce")
    out["volume_ratio_20ma"] = pd.to_numeric(out["volume_ratio_20ma"], errors="coerce")
    return out


def load_poc_cohort(
    path: __import__("pathlib").Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    enrich: bool = True,
    cache_path: __import__("pathlib").Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_derived_features(df)
    df = df[df["tp_target"] == "POC"].copy()
    df = df[df["pair"].astype(str).str.upper().isin(ALLOWED_PAIRS)].copy()
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    df = df.reset_index(drop=True)

    if not enrich:
        return df

    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"])
        if len(cached) == len(df):
            return add_derived_features(cached)

    enriched = enrich_from_ohlcv(df)
    enriched = add_derived_features(enriched)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(cache_path, index=False)
    return enriched
