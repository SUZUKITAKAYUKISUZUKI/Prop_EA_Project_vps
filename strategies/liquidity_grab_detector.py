"""
Liquidity Grab Detector — session/daily pool sweep + immediate failure + reversal close.

Wyckoff 用語（Spring / Upthrust）に依存せず、流動性奪取 → 失敗 → 急速反転のみを抽出する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from strategies.cspa import resolve_cspa_session_type
from strategies.market_utils import pip_size_for_pair

LiquidityPoolType = Literal["DAILY_HIGH", "DAILY_LOW", "SESSION_HIGH", "SESSION_LOW"]
TradeDirection = Literal["BUY", "SELL"]

MIN_SWEEP_ATR = 0.03
STRONG_CLOSE_BODY_RATIO = 0.50
MIN_RECOVERY_RATIO = 0.35


@dataclass(frozen=True)
class GrabDetection:
    is_grabbed: bool
    direction: TradeDirection
    grab_price: float
    sweep_distance_pips: float
    recovery_ratio: float
    grab_strength: float
    liquidity_pool_type: LiquidityPoolType
    trigger_idx: int
    entry_idx: int
    entry_price: float
    stop_loss: float
    take_profit: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_grabbed": self.is_grabbed,
            "direction": self.direction,
            "grab_price": self.grab_price,
            "sweep_distance_pips": self.sweep_distance_pips,
            "recovery_ratio": self.recovery_ratio,
            "grab_strength": self.grab_strength,
            "liquidity_pool_type": self.liquidity_pool_type,
            "trigger_idx": self.trigger_idx,
            "entry_idx": self.entry_idx,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


class LiquidityGrabDetector:
    """M15 トリガー足での流動性奪取（Liquidity Grab）検出。"""

    def __init__(
        self,
        *,
        min_sweep_atr: float = MIN_SWEEP_ATR,
        strong_close_body_ratio: float = STRONG_CLOSE_BODY_RATIO,
        min_recovery_ratio: float = MIN_RECOVERY_RATIO,
        sl_buffer_atr: float = 0.25,
        min_rr: float = 1.5,
    ) -> None:
        self.min_sweep_atr = min_sweep_atr
        self.strong_close_body_ratio = strong_close_body_ratio
        self.min_recovery_ratio = min_recovery_ratio
        self.sl_buffer_atr = sl_buffer_atr
        self.min_rr = min_rr

    def scan(
        self,
        pair: str,
        h1_candles: pd.DataFrame | list[Any],
        m15_candles: pd.DataFrame | list[Any],
    ) -> dict[str, Any]:
        """最新 M15 足で Long/Short Grab を評価（ライブ / 単発チェック用）。"""
        _ = h1_candles
        work = _as_work(m15_candles)
        if len(work) < 3:
            return {"is_grabbed": False}
        idx = len(work) - 2
        pip = pip_size_for_pair(pair)
        long_grab = self.detect_long_grab(work, idx, pair, pip)
        short_grab = self.detect_short_grab(work, idx, pair, pip)
        best: GrabDetection | None = None
        if long_grab is not None and long_grab.is_grabbed:
            best = long_grab
        if short_grab is not None and short_grab.is_grabbed:
            if best is None or short_grab.grab_strength > best.grab_strength:
                best = short_grab
        if best is None:
            return {"is_grabbed": False}
        out = best.as_dict()
        out["is_grabbed"] = True
        return out

    def detect_at_index(
        self,
        work: pd.DataFrame,
        idx: int,
        pair: str,
        atr: float,
    ) -> GrabDetection | None:
        """指定インデックスで Long/Short のうち強い方を返す。"""
        pip = pip_size_for_pair(pair)
        long_grab = self.detect_long_grab(work, idx, pair, pip, atr=atr)
        short_grab = self.detect_short_grab(work, idx, pair, pip, atr=atr)
        candidates = [g for g in (long_grab, short_grab) if g is not None and g.is_grabbed]
        if not candidates:
            return None
        return max(candidates, key=lambda g: g.grab_strength)

    def detect_long_grab(
        self,
        work: pd.DataFrame,
        idx: int,
        pair: str,
        pip: float,
        *,
        atr: float | None = None,
    ) -> GrabDetection | None:
        """
        LONG 条件:
          1. セッション高値・日足高値を更新しない
          2. セッション安値または日足安値を Sweep
          3. 安値下抜け後に即回復
          4. 強い Bull Close
          5. 次足エントリー
        """
        if idx < 1 or idx + 1 >= len(work):
            return None
        atr_val = atr if atr is not None and atr > 0 else _bar_atr(work, idx)
        if atr_val <= 0 or pip <= 0:
            return None

        row = work.iloc[idx]
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_open = float(row["open"])
        bar_close = float(row["close"])
        span = bar_high - bar_low
        if span <= 0:
            return None

        prior_daily_high, prior_daily_low = _prior_daily_extremes(work, idx)
        prior_sess_high, prior_sess_low = _prior_session_extremes(work, idx)
        if prior_daily_high is None or prior_daily_low is None:
            return None

        if bar_high > prior_daily_high + 1e-9:
            return None
        if prior_sess_high is not None and bar_high > prior_sess_high + 1e-9:
            return None

        swept_pools: list[tuple[LiquidityPoolType, float]] = []
        min_sweep = self.min_sweep_atr * atr_val
        if prior_sess_low is not None and bar_low < prior_sess_low - min_sweep:
            swept_pools.append(("SESSION_LOW", prior_sess_low))
        if bar_low < prior_daily_low - min_sweep:
            swept_pools.append(("DAILY_LOW", prior_daily_low))
        if not swept_pools:
            return None

        pool_type, pool_level = min(swept_pools, key=lambda x: x[1])
        sweep_distance = pool_level - bar_low
        if sweep_distance < min_sweep:
            return None

        if bar_close <= pool_level:
            return None

        body_ratio = (bar_close - bar_open) / span
        if bar_close <= bar_open or body_ratio < self.strong_close_body_ratio:
            return None

        recovery_ratio = (bar_close - bar_low) / max(sweep_distance, 1e-9)
        if recovery_ratio < self.min_recovery_ratio:
            return None

        entry_idx = idx + 1
        entry_price = float(work.iloc[entry_idx]["open"])
        stop_loss = bar_low - self.sl_buffer_atr * atr_val
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
        take_profit = entry_price + self.min_rr * risk
        sweep_pips = sweep_distance / pip
        grab_strength = _grab_strength(recovery_ratio, sweep_pips, body_ratio)

        return GrabDetection(
            is_grabbed=True,
            direction="BUY",
            grab_price=round(bar_low, 6),
            sweep_distance_pips=round(sweep_pips, 4),
            recovery_ratio=round(recovery_ratio, 4),
            grab_strength=round(grab_strength, 4),
            liquidity_pool_type=pool_type,
            trigger_idx=idx,
            entry_idx=entry_idx,
            entry_price=round(entry_price, 6),
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
        )

    def detect_short_grab(
        self,
        work: pd.DataFrame,
        idx: int,
        pair: str,
        pip: float,
        *,
        atr: float | None = None,
    ) -> GrabDetection | None:
        """
        SHORT 条件:
          1. セッション安値・日足安値を更新しない
          2. セッション高値または日足高値を Sweep
          3. 高値突破後に即失敗
          4. 強い Bear Close
          5. 次足エントリー
        """
        if idx < 1 or idx + 1 >= len(work):
            return None
        atr_val = atr if atr is not None and atr > 0 else _bar_atr(work, idx)
        if atr_val <= 0 or pip <= 0:
            return None

        row = work.iloc[idx]
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_open = float(row["open"])
        bar_close = float(row["close"])
        span = bar_high - bar_low
        if span <= 0:
            return None

        prior_daily_high, prior_daily_low = _prior_daily_extremes(work, idx)
        prior_sess_high, prior_sess_low = _prior_session_extremes(work, idx)
        if prior_daily_high is None or prior_daily_low is None:
            return None

        if bar_low < prior_daily_low - 1e-9:
            return None
        if prior_sess_low is not None and bar_low < prior_sess_low - 1e-9:
            return None

        swept_pools: list[tuple[LiquidityPoolType, float]] = []
        min_sweep = self.min_sweep_atr * atr_val
        if prior_sess_high is not None and bar_high > prior_sess_high + min_sweep:
            swept_pools.append(("SESSION_HIGH", prior_sess_high))
        if bar_high > prior_daily_high + min_sweep:
            swept_pools.append(("DAILY_HIGH", prior_daily_high))
        if not swept_pools:
            return None

        pool_type, pool_level = max(swept_pools, key=lambda x: x[1])
        sweep_distance = bar_high - pool_level
        if sweep_distance < min_sweep:
            return None

        if bar_close >= pool_level:
            return None

        body_ratio = (bar_open - bar_close) / span
        if bar_close >= bar_open or body_ratio < self.strong_close_body_ratio:
            return None

        recovery_ratio = (bar_high - bar_close) / max(sweep_distance, 1e-9)
        if recovery_ratio < self.min_recovery_ratio:
            return None

        entry_idx = idx + 1
        entry_price = float(work.iloc[entry_idx]["open"])
        stop_loss = bar_high + self.sl_buffer_atr * atr_val
        risk = stop_loss - entry_price
        if risk <= 0:
            return None
        take_profit = entry_price - self.min_rr * risk
        sweep_pips = sweep_distance / pip
        grab_strength = _grab_strength(recovery_ratio, sweep_pips, body_ratio)

        return GrabDetection(
            is_grabbed=True,
            direction="SELL",
            grab_price=round(bar_high, 6),
            sweep_distance_pips=round(sweep_pips, 4),
            recovery_ratio=round(recovery_ratio, 4),
            grab_strength=round(grab_strength, 4),
            liquidity_pool_type=pool_type,
            trigger_idx=idx,
            entry_idx=entry_idx,
            entry_price=round(entry_price, 6),
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
        )


def _as_work(candles: pd.DataFrame | list[Any]) -> pd.DataFrame:
    if isinstance(candles, pd.DataFrame):
        work = candles.sort_values("datetime").reset_index(drop=True).copy()
        work["datetime"] = pd.to_datetime(work["datetime"])
        if "volume" not in work.columns:
            work["volume"] = 0.0
        return work
    raise TypeError("candles must be a pandas DataFrame")


def _bar_atr(work: pd.DataFrame, idx: int, period: int = 14) -> float:
    from strategies.market_utils import compute_atr

    clipped = work.iloc[: idx + 1]
    if len(clipped) < period + 1:
        return 0.0
    atr = compute_atr(clipped, period)
    if idx >= len(atr):
        return 0.0
    val = float(atr.iloc[idx])
    return val if val > 0 else 0.0


def _prior_daily_extremes(work: pd.DataFrame, idx: int) -> tuple[float | None, float | None]:
    if idx < 1:
        return None, None
    day_norm = pd.to_datetime(work["datetime"]).dt.normalize()
    current_day = day_norm.iloc[idx]
    prior = work.iloc[:idx]
    prior = prior.loc[day_norm.iloc[:idx] == current_day]
    if prior.empty:
        return None, None
    return float(prior["high"].max()), float(prior["low"].min())


def _prior_session_extremes(work: pd.DataFrame, idx: int) -> tuple[float | None, float | None]:
    if idx < 1:
        return None, None
    ts = pd.Timestamp(work.iloc[idx]["datetime"])
    session = resolve_cspa_session_type(ts)
    day_norm = pd.to_datetime(work["datetime"]).dt.normalize()
    current_day = day_norm.iloc[idx]
    prior = work.iloc[:idx]
    prior = prior.loc[day_norm.iloc[:idx] == current_day]
    if prior.empty:
        return None, None
    mask = prior["datetime"].apply(lambda t: _bar_in_session(pd.Timestamp(t), session))
    seg = prior.loc[mask] if mask.any() else prior
    if seg.empty:
        return None, None
    return float(seg["high"].max()), float(seg["low"].min())


def _bar_in_session(ts: pd.Timestamp, session: str) -> bool:
    hour = int(ts.hour)
    if session == "LONDON":
        return 8 <= hour < 17
    if session == "NY":
        return 13 <= hour < 22
    if session == "ASIA":
        return hour < 8
    return True


def _grab_strength(recovery_ratio: float, sweep_pips: float, body_ratio: float) -> float:
    sweep_score = min(sweep_pips / 3.0, 1.0)
    return 0.45 * recovery_ratio + 0.35 * sweep_score + 0.20 * body_ratio


__all__ = [
    "GrabDetection",
    "LiquidityGrabDetector",
    "LiquidityPoolType",
]
