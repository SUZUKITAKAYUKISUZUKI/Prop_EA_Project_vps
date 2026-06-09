"""
strategies/htf_trend_analyzer.py — 上位足トレンド共通インフラ

MA200 と直近ダウ理論（スイング高値・安値の HH/HL / LH/LL）を組み合わせ、
BULL / BEAR / NEUTRAL を返す。全戦略から import 可能な純粋 pandas モジュール。

上位足は H1 OHLCV をリサンプルして構築（デフォルト H4）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TrendDirection = Literal["BULL", "BEAR", "NEUTRAL"]

HTF_RESAMPLE_RULE = os.getenv("HTF_TREND_RESAMPLE_RULE", "4h")
HTF_MA_PERIOD = int(os.getenv("HTF_MA_PERIOD", "200"))
HTF_SWING_LOOKBACK = int(os.getenv("HTF_SWING_LOOKBACK", "2"))
HTF_BAR_HOURS = int(os.getenv("HTF_BAR_HOURS", "4"))


@dataclass(frozen=True)
class HtfTrendResult:
    """上位足トレンド判定結果。"""

    direction: TrendDirection
    ma_bias: TrendDirection
    dow_bias: TrendDirection
    ma200: float | None
    close: float | None
    htf_bars_used: int


def resample_to_htf(h1_df: pd.DataFrame, rule: str = HTF_RESAMPLE_RULE) -> pd.DataFrame:
    """H1（または下位足）OHLCV を上位足へリサンプル。"""
    from strategies.bt_ohlcv import BtOhlcvFrame, resample_bars_ns

    if isinstance(h1_df, BtOhlcvFrame):
        if h1_df.empty:
            return BtOhlcvFrame.make_empty()
        bar_ns = int(pd.Timedelta(rule).value)
        return resample_bars_ns(h1_df, bar_ns)

    if h1_df is None or h1_df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(h1_df.columns):
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    indexed = h1_df.sort_values("datetime").set_index("datetime")
    resampled = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in indexed.columns else "first",
        }
    )
    return resampled.dropna(subset=["open"]).reset_index()


def clip_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """評価時点までのバーのみ残す（未来データ排除）。"""
    from strategies.bt_ohlcv import BtOhlcvFrame, asof_end_index, normalize_ts_ns

    if isinstance(df, BtOhlcvFrame):
        if df.empty:
            return df
        end = asof_end_index(df.arrays, normalize_ts_ns(as_of))
        return df.slice_end(end)

    if df is None or df.empty:
        return df
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    dt = pd.to_datetime(df["datetime"])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return df.loc[dt <= ts].reset_index(drop=True)


def drop_incomplete_htf_bar(htf_df: pd.DataFrame, as_of: pd.Timestamp, bar_hours: int = HTF_BAR_HOURS) -> pd.DataFrame:
    """未確定の最終上位足バーを除外。"""
    from strategies.bt_ohlcv import BtOhlcvFrame, asof_end_index, normalize_ts_ns

    if isinstance(htf_df, BtOhlcvFrame):
        if htf_df.empty:
            return htf_df
        ts = pd.Timestamp(as_of)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        cutoff_ns = normalize_ts_ns(ts - pd.Timedelta(hours=bar_hours))
        end = asof_end_index(htf_df.arrays, cutoff_ns)
        return htf_df.slice_end(end)

    if htf_df is None or htf_df.empty:
        return htf_df
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    cutoff = ts - pd.Timedelta(hours=bar_hours)
    dt = pd.to_datetime(htf_df["datetime"])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return htf_df.loc[dt <= cutoff].reset_index(drop=True)


def classify_ma200_bias(
    htf_df: pd.DataFrame,
    ma_period: int = HTF_MA_PERIOD,
) -> tuple[TrendDirection, float | None, float | None]:
    """終値 vs MA200 による方向バイアス。"""
    from strategies.bt_ohlcv import BtOhlcvFrame

    if isinstance(htf_df, BtOhlcvFrame):
        arr = htf_df.arrays
        if arr.length < ma_period:
            return "NEUTRAL", None, None
        last_close = float(arr.close[-1])
        last_ma = float(np.mean(arr.close[-ma_period:]))
        if not np.isfinite(last_ma):
            return "NEUTRAL", last_ma, last_close
        if last_close > last_ma:
            return "BULL", last_ma, last_close
        if last_close < last_ma:
            return "BEAR", last_ma, last_close
        return "NEUTRAL", last_ma, last_close

    if htf_df is None or len(htf_df) < ma_period:
        return "NEUTRAL", None, None

    ma_series = htf_df["close"].rolling(ma_period, min_periods=ma_period).mean()
    last_close = float(htf_df["close"].iloc[-1])
    last_ma = float(ma_series.iloc[-1])
    if pd.isna(last_ma):
        return "NEUTRAL", last_ma, last_close

    if last_close > last_ma:
        return "BULL", last_ma, last_close
    if last_close < last_ma:
        return "BEAR", last_ma, last_close
    return "NEUTRAL", last_ma, last_close


def find_confirmed_swings(
    htf_df: pd.DataFrame,
    lookback: int = HTF_SWING_LOOKBACK,
) -> tuple[list[float], list[float]]:
    """
    確定スイング高値・安値を抽出（左右 lookback 本より極値）。

    末尾 lookback 本は未確定のため評価対象外。
    """
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    from strategies.bt_ohlcv import BtOhlcvFrame

    if isinstance(htf_df, BtOhlcvFrame):
        highs = htf_df.arrays.high
        lows = htf_df.arrays.low
        n = htf_df.arrays.length
    else:
        if htf_df is None or len(htf_df) < lookback * 2 + 1:
            return swing_highs, swing_lows
        highs = htf_df["high"].astype(float).values
        lows = htf_df["low"].astype(float).values
        n = len(htf_df)

    if n < lookback * 2 + 1:
        return swing_highs, swing_lows

    last_confirmable = n - lookback

    for i in range(lookback, last_confirmable):
        left_h = highs[i - lookback : i]
        right_h = highs[i + 1 : i + lookback + 1]
        if highs[i] >= left_h.max() and highs[i] >= right_h.max():
            swing_highs.append(float(highs[i]))

        left_l = lows[i - lookback : i]
        right_l = lows[i + 1 : i + lookback + 1]
        if lows[i] <= left_l.min() and lows[i] <= right_l.min():
            swing_lows.append(float(lows[i]))

    return swing_highs, swing_lows


def classify_dow_structure(
    swing_highs: list[float],
    swing_lows: list[float],
) -> TrendDirection:
    """直近2スイングの HH/HL → BULL、LH/LL → BEAR、それ以外 NEUTRAL。"""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "NEUTRAL"

    prev_high, last_high = swing_highs[-2], swing_highs[-1]
    prev_low, last_low = swing_lows[-2], swing_lows[-1]

    higher_high = last_high > prev_high
    higher_low = last_low > prev_low
    lower_high = last_high < prev_high
    lower_low = last_low < prev_low

    if higher_high and higher_low:
        return "BULL"
    if lower_high and lower_low:
        return "BEAR"
    return "NEUTRAL"


def combine_trend_signals(ma_bias: TrendDirection, dow_bias: TrendDirection) -> TrendDirection:
    """MA200 とダウ構造が一致した場合のみ方向性を確定（保守的）。"""
    if ma_bias == "BULL" and dow_bias == "BULL":
        return "BULL"
    if ma_bias == "BEAR" and dow_bias == "BEAR":
        return "BEAR"
    return "NEUTRAL"


def analyze_htf_trend(
    h1_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    htf_df: pd.DataFrame | None = None,
    resample_rule: str = HTF_RESAMPLE_RULE,
    ma_period: int = HTF_MA_PERIOD,
    swing_lookback: int = HTF_SWING_LOOKBACK,
    bar_hours: int = HTF_BAR_HOURS,
) -> HtfTrendResult:
    """
    評価時点の上位足トレンドを判定する。

    Args:
        h1_df: 構造足 OHLCV（通常 H1）。htf_df 未指定時はここからリサンプル。
        htf_df: ネイティブ H4 等の上位足 CSV（指定時はリサンプルをスキップ）
        as_of: セットアップ評価タイムスタンプ（未来バー不使用）
    """
    empty = HtfTrendResult("NEUTRAL", "NEUTRAL", "NEUTRAL", None, None, 0)

    if htf_df is not None and not htf_df.empty:
        clipped = clip_as_of(htf_df, as_of)
        if clipped.empty:
            return empty
        htf = drop_incomplete_htf_bar(clipped, as_of, bar_hours)
    else:
        if h1_df is None or h1_df.empty:
            return empty
        clipped = clip_as_of(h1_df, as_of)
        if clipped.empty:
            return empty
        htf = resample_to_htf(clipped, resample_rule)
        htf = drop_incomplete_htf_bar(htf, as_of, bar_hours)

    if htf.empty:
        return empty

    ma_bias, ma200, close = classify_ma200_bias(htf, ma_period)
    swing_highs, swing_lows = find_confirmed_swings(htf, swing_lookback)
    dow_bias = classify_dow_structure(swing_highs, swing_lows)
    direction = combine_trend_signals(ma_bias, dow_bias)

    return HtfTrendResult(
        direction=direction,
        ma_bias=ma_bias,
        dow_bias=dow_bias,
        ma200=ma200,
        close=close,
        htf_bars_used=len(htf),
    )


@dataclass
class HtfTrendLookup:
    """Backtest 向け HTF トレンドキャッシュ（H1 終端インデックス単位でリサンプル結果を再利用）。"""

    ma_period: int
    swing_lookback: int
    bar_hours: int
    resample_rule: str
    _h1: pd.DataFrame
    _h1_dt_index: pd.DatetimeIndex
    _native_htf: bool
    _htf_full: pd.DataFrame
    _htf_dt_index: pd.DatetimeIndex
    _htf_cache: dict[int, pd.DataFrame]
    _empty: HtfTrendResult

    @staticmethod
    def _coerce_datetime_index(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))

    @classmethod
    def from_h1(
        cls,
        h1_df: pd.DataFrame,
        *,
        htf_df: pd.DataFrame | None = None,
        resample_rule: str = HTF_RESAMPLE_RULE,
        ma_period: int = HTF_MA_PERIOD,
        swing_lookback: int = HTF_SWING_LOOKBACK,
        bar_hours: int = HTF_BAR_HOURS,
    ) -> HtfTrendLookup:
        empty = HtfTrendResult("NEUTRAL", "NEUTRAL", "NEUTRAL", None, None, 0)
        if h1_df is None or h1_df.empty:
            return cls(
                ma_period=ma_period,
                swing_lookback=swing_lookback,
                bar_hours=bar_hours,
                resample_rule=resample_rule,
                _h1=pd.DataFrame(columns=["datetime", "open", "high", "low", "close"]),
                _h1_dt_index=pd.DatetimeIndex([]),
                _native_htf=False,
                _htf_full=pd.DataFrame(),
                _htf_dt_index=pd.DatetimeIndex([]),
                _htf_cache={},
                _empty=empty,
            )

        h1 = h1_df.sort_values("datetime").reset_index(drop=True)
        h1["datetime"] = pd.to_datetime(h1["datetime"])
        h1_dt_index = cls._coerce_datetime_index(h1["datetime"])
        native = htf_df is not None and not htf_df.empty
        if native:
            htf = htf_df.sort_values("datetime").reset_index(drop=True)
            htf["datetime"] = pd.to_datetime(htf["datetime"])
            htf_dt_index = cls._coerce_datetime_index(htf["datetime"])
        else:
            htf = pd.DataFrame()
            htf_dt_index = pd.DatetimeIndex([])

        return cls(
            ma_period=ma_period,
            swing_lookback=swing_lookback,
            bar_hours=bar_hours,
            resample_rule=resample_rule,
            _h1=h1,
            _h1_dt_index=h1_dt_index,
            _native_htf=native,
            _htf_full=htf,
            _htf_dt_index=htf_dt_index,
            _htf_cache={},
            _empty=empty,
        )

    def _htf_slice(self, as_of: pd.Timestamp) -> pd.DataFrame:
        ts = pd.Timestamp(as_of)
        cutoff = ts - pd.Timedelta(hours=self.bar_hours)
        if self._native_htf:
            end = int(self._htf_dt_index.searchsorted(cutoff, side="right")) - 1
            if end < 0:
                return pd.DataFrame()
            return self._htf_full.iloc[: end + 1]

        h1_end = int(self._h1_dt_index.searchsorted(ts, side="right")) - 1
        if h1_end < 0:
            return pd.DataFrame()

        cached = self._htf_cache.get(h1_end)
        if cached is None:
            cached = resample_to_htf(self._h1.iloc[: h1_end + 1], self.resample_rule)
            self._htf_cache[h1_end] = cached
        return drop_incomplete_htf_bar(cached, ts, self.bar_hours)

    def at(self, as_of: pd.Timestamp) -> HtfTrendResult:
        htf = self._htf_slice(as_of)
        if htf.empty:
            return self._empty

        ma_bias, ma200, close = classify_ma200_bias(htf, self.ma_period)
        swing_highs, swing_lows = find_confirmed_swings(htf, self.swing_lookback)
        dow_bias = classify_dow_structure(swing_highs, swing_lows)
        direction = combine_trend_signals(ma_bias, dow_bias)
        return HtfTrendResult(
            direction=direction,
            ma_bias=ma_bias,
            dow_bias=dow_bias,
            ma200=ma200,
            close=close,
            htf_bars_used=len(htf),
        )


def build_htf_trend_lookup(
    h1_df: pd.DataFrame,
    *,
    htf_df: pd.DataFrame | None = None,
    resample_rule: str = HTF_RESAMPLE_RULE,
    ma_period: int = HTF_MA_PERIOD,
    swing_lookback: int = HTF_SWING_LOOKBACK,
    bar_hours: int = HTF_BAR_HOURS,
) -> HtfTrendLookup:
    """``detect_setups`` 等のバーループ向け HTF キャッシュを構築する。"""
    return HtfTrendLookup.from_h1(
        h1_df,
        htf_df=htf_df,
        resample_rule=resample_rule,
        ma_period=ma_period,
        swing_lookback=swing_lookback,
        bar_hours=bar_hours,
    )


def is_counter_trend(trade_direction: str, htf_trend: TrendDirection) -> bool:
    """上位足トレンドと逆行するエントリー方向か。"""
    if htf_trend == "NEUTRAL":
        return False

    direction = trade_direction.strip().upper()
    if htf_trend == "BULL" and direction in ("SELL", "SHORT"):
        return True
    if htf_trend == "BEAR" and direction in ("BUY", "LONG"):
        return True
    return False
