"""
strategies/archive/fvg_fill.py — FVG Fill 平均回帰（旧 Strategy C）

ARCHIVED 2026-06-09: CSPA を Strategy B として正式採用。FVG は WFT 後に本番・標準 BT から外した。参照・再検討用。

Bullish FVG (FVG_LONG / BUY) と Bearish FVG (FVG_SHORT / SELL) を M15/H1 で検出し、
Gemini L4 監査通過後に執行する。L5 追跡は entry timestamp 同期（bar_index 流用禁止）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import analyze_htf_trend, clip_as_of, is_counter_trend
from strategies.market_utils import (
    PIP_SIZE,
    SMTFeatures,
    calc_smt_features,
    compute_atr,
    pip_size_for_pair,
    positional_index as _positional_index,
)

SETUP_TYPE = "FVG_FILL"
MIN_RR = 2.0
# §5.2.2 高確信度ロット帯 (1.4x) の参照閾値。L4 足切りは platform 共通の 40 未満。
FVG_HIGH_CONFIDENCE_MIN = int(os.getenv("FVG_LLM_MIN_CONFIDENCE", "85"))
FVG_EXECUTE_MIN_CONFIDENCE = FVG_HIGH_CONFIDENCE_MIN  # backward-compat alias
ORDER_BLOCK_LOOKBACK = 20
ORDER_BLOCK_MAX_DISTANCE_PIPS = 25.0
FVG_SL_BUFFER_PIPS = float(os.getenv("FVG_SL_BUFFER_PIPS", "2.0"))
VOLATILITY_LOOKBACK_DAYS = 20
HTF_TREND_MISMATCH_TAG = "HTF_TREND_MISMATCH"
CAUTION_HTF_COUNTER_TAG = "CAUTION_HTF_COUNTER"
# L2 足切り（FVG 専用。L2_SPEED_MIN_SCORE=10 から緩和し 6〜10 帯の機会損失を回収）
FVG_L2_MIN_SCORE = int(os.getenv("FVG_L2_MIN_SCORE", "6"))
FVG_HTF_COUNTER_MULTIPLIER = float(os.getenv("FVG_HTF_COUNTER_MULTIPLIER", "0.7"))
FVG_HTF_NEUTRAL_MULTIPLIER = float(os.getenv("FVG_HTF_NEUTRAL_MULTIPLIER", "0.85"))
FVG_SHORT_FILL_FRACTION = float(os.getenv("FVG_SHORT_FILL_FRACTION", "0.5"))
FVG_ENABLE_LONG = os.getenv("FVG_ENABLE_LONG", "1").strip().lower() in ("1", "true", "yes", "on")
FVG_ENABLE_SHORT = os.getenv("FVG_ENABLE_SHORT", "1").strip().lower() in ("1", "true", "yes", "on")
FVG_SIDE_LONG = "LONG"
FVG_SIDE_SHORT = "SHORT"


@dataclass(frozen=True)
class FvgFillConfig:
    min_fvg_pips: float = 3.0
    max_fvg_pips: float = 40.0
    max_setups_per_day: int = 1


def load_fvg_config() -> FvgFillConfig:
    return FvgFillConfig(
        min_fvg_pips=float(os.getenv("FVG_MIN_PIPS", "3.0")),
        max_fvg_pips=float(os.getenv("FVG_MAX_PIPS", "40.0")),
        max_setups_per_day=int(os.getenv("FVG_MAX_SETUPS_PER_DAY", "1")),
    )


@dataclass
class FvgFillSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: str
    fvg_top: float
    fvg_bottom: float
    fvg_size_pips: float
    entry_price: float
    stop_loss: float
    take_profit: float
    sweep_distance: float
    atr: float
    bar_index: int
    nearby_order_block_present: bool
    nearby_order_block_distance_pips: float
    volatility_20d: float
    current_session: str
    fvg_side: str = FVG_SIDE_LONG


def resolve_current_session(hour: int) -> str:
    if 0 <= hour <= 6:
        return "ASIA"
    if 15 <= hour <= 20:
        return "LONDON"
    if hour >= 21:
        return "NY"
    return "OFF_SESSION"


def _ensure_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work = df.sort_values("datetime").copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    if "hour" not in work.columns:
        work["hour"] = work["datetime"].dt.hour
    if "date" not in work.columns:
        work["date"] = work["datetime"].dt.date
    return work.reset_index(drop=True)


def detect_bullish_fvg_gap(
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
) -> tuple[float, float] | None:
    """
    Bullish FVG (BUY): 3本前の安値が1本前の高値より上に離れている。
    条件: lows[idx-3] > highs[idx-1]
    ゾーン: fvg_bottom = highs[idx-1], fvg_top = lows[idx-3]
    """
    if idx < 3:
        return None
    if float(lows[idx - 3]) <= float(highs[idx - 1]):
        return None
    fvg_bottom = float(highs[idx - 1])
    fvg_top = float(lows[idx - 3])
    if fvg_top <= fvg_bottom:
        return None
    return fvg_bottom, fvg_top


def _bar_intersects_fvg(bar: pd.Series, fvg_bottom: float, fvg_top: float) -> bool:
    return float(bar["low"]) <= fvg_top and float(bar["high"]) >= fvg_bottom


def detect_bearish_fvg_gap(
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
) -> tuple[float, float] | None:
    """
    Bearish FVG (SELL): Bullish の数学的逆 — 3本前の高値が1本前の安値より下に離れている。
    条件: highs[idx-3] < lows[idx-1]
    ゾーン: fvg_bottom = highs[idx-3] (Bar1 High / ギャップ下端),
            fvg_top = lows[idx-1] (Bar3 Low / ギャップ上端)
    """
    if idx < 3:
        return None
    if float(highs[idx - 3]) >= float(lows[idx - 1]):
        return None
    fvg_bottom = float(highs[idx - 3])
    fvg_top = float(lows[idx - 1])
    if fvg_top <= fvg_bottom:
        return None
    return fvg_bottom, fvg_top


def _bar_intersects_fvg_short_fill_zone(
    bar: pd.Series,
    fvg_bottom: float,
    fvg_top: float,
    fill_fraction: float = FVG_SHORT_FILL_FRACTION,
) -> bool:
    """Bearish fill: 下端〜 fill_fraction プレミアム（ゾーン下半分）へのプルバック。"""
    if fvg_top <= fvg_bottom:
        return False
    fill_ceiling = fvg_bottom + fill_fraction * (fvg_top - fvg_bottom)
    return float(bar["low"]) <= fill_ceiling and float(bar["high"]) >= fvg_bottom


def _compute_volatility_20d(h1_df: pd.DataFrame, current_ts: pd.Timestamp, pip_size: float) -> float:
    clipped = clip_as_of(_ensure_bars(h1_df), current_ts)
    if clipped.empty:
        return 0.0
    atr_series = compute_atr(clipped)
    valid = atr_series.dropna()
    if valid.empty:
        return 0.0
    tail = valid.tail(VOLATILITY_LOOKBACK_DAYS * 24)
    return float(tail.mean()) / pip_size if len(tail) else 0.0


def _detect_nearby_order_block(
    structure_df: pd.DataFrame,
    bar_index: int,
    fvg_bottom: float,
    pip_size: float,
) -> tuple[bool, float]:
    if bar_index < ORDER_BLOCK_LOOKBACK:
        return False, 999.0
    window = structure_df.iloc[max(0, bar_index - ORDER_BLOCK_LOOKBACK) : bar_index]
    if window.empty:
        return False, 999.0

    ref = fvg_bottom
    swings = window["low"].astype(float)
    block = float(swings.min())
    distance_pips = abs(ref - block) / pip_size
    present = block <= ref and distance_pips <= ORDER_BLOCK_MAX_DISTANCE_PIPS
    return present, distance_pips


def _compute_long_sl_tp(
    entry: float,
    fvg_bottom: float,
    pip_size: float,
) -> tuple[float, float] | None:
    """LONG: SL = fvg_bottom - buffer, TP = entry + RR * risk。"""
    sl_buffer = pip_size * FVG_SL_BUFFER_PIPS
    stop_loss = fvg_bottom - sl_buffer
    risk = entry - stop_loss
    if risk <= 0:
        return None
    take_profit = entry + MIN_RR * risk
    return stop_loss, take_profit


def _compute_short_sl_tp(
    entry: float,
    fvg_top: float,
    pip_size: float,
) -> tuple[float, float] | None:
    """SHORT: SL = fvg_top + buffer (entry より上), TP = entry - RR * risk。"""
    sl_buffer = pip_size * FVG_SL_BUFFER_PIPS
    stop_loss = fvg_top + sl_buffer
    risk = stop_loss - entry
    if risk <= 0:
        return None
    take_profit = entry - MIN_RR * risk
    return stop_loss, take_profit


def _detect_nearby_order_block_short(
    structure_df: pd.DataFrame,
    bar_index: int,
    fvg_top: float,
    pip_size: float,
) -> tuple[bool, float]:
    """Bearish: fvg_top 付近の supply (直近スイング高値) を探索。"""
    if bar_index < ORDER_BLOCK_LOOKBACK:
        return False, 999.0
    window = structure_df.iloc[max(0, bar_index - ORDER_BLOCK_LOOKBACK) : bar_index]
    if window.empty:
        return False, 999.0

    ref = fvg_top
    block = float(window["high"].astype(float).max())
    distance_pips = abs(block - ref) / pip_size
    present = block >= ref and distance_pips <= ORDER_BLOCK_MAX_DISTANCE_PIPS
    return present, distance_pips


def _build_fvg_setup(
    bar: pd.Series,
    bar_index: int,
    pair_name: str,
    fvg_bottom: float,
    fvg_top: float,
    atr_val: float,
    pip_size: float,
    structure_df: pd.DataFrame,
    h1_df: pd.DataFrame | None,
) -> FvgFillSetup | None:
    entry = float(bar["close"])
    zone_height = fvg_top - fvg_bottom
    levels = _compute_long_sl_tp(entry, fvg_bottom, pip_size)
    if levels is None:
        return None
    stop_loss, take_profit = levels

    ts = pd.Timestamp(bar["datetime"])
    fvg_size_pips = zone_height / pip_size
    h1_ref = h1_df if h1_df is not None else structure_df
    vol_20d = _compute_volatility_20d(h1_ref, ts, pip_size)
    ob_present, ob_dist = _detect_nearby_order_block(
        structure_df, bar_index, fvg_bottom, pip_size
    )

    return FvgFillSetup(
        timestamp=ts,
        pair=pair_name,
        direction="BUY",
        fvg_side=FVG_SIDE_LONG,
        fvg_top=fvg_top,
        fvg_bottom=fvg_bottom,
        fvg_size_pips=fvg_size_pips,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        sweep_distance=fvg_top - fvg_bottom,
        atr=atr_val,
        bar_index=bar_index,
        nearby_order_block_present=ob_present,
        nearby_order_block_distance_pips=ob_dist,
        volatility_20d=vol_20d,
        current_session=resolve_current_session(int(bar["hour"])),
    )


def _build_fvg_short_setup(
    bar: pd.Series,
    bar_index: int,
    pair_name: str,
    fvg_bottom: float,
    fvg_top: float,
    atr_val: float,
    pip_size: float,
    structure_df: pd.DataFrame,
    h1_df: pd.DataFrame | None,
) -> FvgFillSetup | None:
    entry = float(bar["close"])
    levels = _compute_short_sl_tp(entry, fvg_top, pip_size)
    if levels is None:
        return None
    stop_loss, take_profit = levels
    if not (entry < stop_loss and take_profit < entry):
        return None

    ts = pd.Timestamp(bar["datetime"])
    zone_height = fvg_top - fvg_bottom
    fvg_size_pips = zone_height / pip_size
    h1_ref = h1_df if h1_df is not None else structure_df
    vol_20d = _compute_volatility_20d(h1_ref, ts, pip_size)
    ob_present, ob_dist = _detect_nearby_order_block_short(
        structure_df, bar_index, fvg_top, pip_size
    )

    return FvgFillSetup(
        timestamp=ts,
        pair=pair_name,
        direction="SELL",
        fvg_side=FVG_SIDE_SHORT,
        fvg_top=fvg_top,
        fvg_bottom=fvg_bottom,
        fvg_size_pips=fvg_size_pips,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        sweep_distance=fvg_top - fvg_bottom,
        atr=atr_val,
        bar_index=bar_index,
        nearby_order_block_present=ob_present,
        nearby_order_block_distance_pips=ob_dist,
        volatility_20d=vol_20d,
        current_session=resolve_current_session(int(bar["hour"])),
    )


def _scan_fvg_setups(
    exec_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    pair_name: str,
    cfg: FvgFillConfig,
    *,
    side: str,
    progress_hook: Callable[[int, int], None] | None = None,
    h1_df: pd.DataFrame | None = None,
) -> list[FvgFillSetup]:
    pip_size = pip_size_for_pair(pair_name)
    atr_series = compute_atr(structure_df)
    highs = exec_df["high"].to_numpy(dtype=float)
    lows = exec_df["low"].to_numpy(dtype=float)

    setups: list[FvgFillSetup] = []
    emitted_dates: dict[date, int] = {}
    scan_total = max(len(exec_df) - 1, 1)
    is_long = side == FVG_SIDE_LONG

    for i in range(3, len(exec_df)):
        if progress_hook is not None:
            progress_hook(i, scan_total)
        bar = exec_df.iloc[i]
        session_date = pd.Timestamp(bar["datetime"]).date()
        if emitted_dates.get(session_date, 0) >= cfg.max_setups_per_day:
            continue

        if is_long:
            gap = detect_bullish_fvg_gap(highs, lows, i)
            if gap is None:
                continue
            fvg_bottom, fvg_top = gap
            if not _bar_intersects_fvg(bar, fvg_bottom, fvg_top):
                continue
            build_fn = _build_fvg_setup
        else:
            gap = detect_bearish_fvg_gap(highs, lows, i)
            if gap is None:
                continue
            fvg_bottom, fvg_top = gap
            if not _bar_intersects_fvg_short_fill_zone(bar, fvg_bottom, fvg_top):
                continue
            build_fn = _build_fvg_short_setup

        fvg_pips = (fvg_top - fvg_bottom) / pip_size
        if fvg_pips < cfg.min_fvg_pips or fvg_pips > cfg.max_fvg_pips:
            continue

        struct_idx = structure_df.index[structure_df["datetime"] == bar["datetime"]]
        bar_pos = _positional_index(structure_df, struct_idx[0]) if len(struct_idx) else i
        atr_val = float(atr_series.iloc[bar_pos]) if bar_pos < len(atr_series) and pd.notna(
            atr_series.iloc[bar_pos]
        ) else max(float(bar["high"] - bar["low"]), pip_size * 10)

        setup = build_fn(
            bar,
            bar_pos,
            pair_name,
            fvg_bottom,
            fvg_top,
            atr_val,
            pip_size,
            structure_df,
            h1_df,
        )
        if setup is not None:
            setups.append(setup)
            emitted_dates[session_date] = emitted_dates.get(session_date, 0) + 1

    return setups


def detect_fvg_fill_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    config: FvgFillConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> list[FvgFillSetup]:
    """M15/H1: Bullish FVG のみ検出（FVG_LONG / BUY、ルックアヘッドなし）。"""
    return detect_fvg_long_fill_setups(df, pair_name, h1_df, config, progress_hook)


def detect_fvg_long_fill_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    config: FvgFillConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> list[FvgFillSetup]:
    """FVG_LONG (Bullish gap → BUY) のみ。"""
    cfg = config or load_fvg_config()
    exec_df = _ensure_bars(df)
    structure_df = _ensure_bars(h1_df if h1_df is not None else df)
    if exec_df.empty or structure_df.empty:
        return []
    return _scan_fvg_setups(
        exec_df,
        structure_df,
        pair_name,
        cfg,
        side=FVG_SIDE_LONG,
        progress_hook=progress_hook,
        h1_df=h1_df,
    )


def detect_fvg_short_fill_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    config: FvgFillConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> list[FvgFillSetup]:
    """FVG_SHORT (Bearish gap → SELL) のみ。"""
    cfg = config or load_fvg_config()
    exec_df = _ensure_bars(df)
    structure_df = _ensure_bars(h1_df if h1_df is not None else df)
    if exec_df.empty or structure_df.empty:
        return []
    return _scan_fvg_setups(
        exec_df,
        structure_df,
        pair_name,
        cfg,
        side=FVG_SIDE_SHORT,
        progress_hook=progress_hook,
        h1_df=h1_df,
    )


def detect_fvg_all_fill_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    config: FvgFillConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
    *,
    enable_long: bool | None = None,
    enable_short: bool | None = None,
) -> list[FvgFillSetup]:
    """FVG_LONG + FVG_SHORT をタイムスタンプ順にマージ。"""
    use_long = FVG_ENABLE_LONG if enable_long is None else enable_long
    use_short = FVG_ENABLE_SHORT if enable_short is None else enable_short
    merged: list[FvgFillSetup] = []
    if use_long:
        merged.extend(
            detect_fvg_long_fill_setups(
                df, pair_name, h1_df, config, progress_hook=progress_hook
            )
        )
    if use_short:
        merged.extend(
            detect_fvg_short_fill_setups(
                df, pair_name, h1_df, config, progress_hook=progress_hook
            )
        )
    merged.sort(key=lambda s: s.timestamp)
    return merged


def is_htf_trend_mismatch(trade_direction: str, htf_trend: str) -> bool:
    """HTF 方向性判定（BUY/SELL 対称）。逆方向はロット縮小へ。"""
    return is_counter_trend(trade_direction, htf_trend)  # type: ignore[arg-type]


def resolve_fvg_htf_lot_multiplier(
    trade_direction: str,
    htf_trend: str,
    *,
    counter_multiplier: float | None = None,
    neutral_multiplier: float | None = None,
) -> tuple[bool, float]:
    """
    HTF ソフトフィルター: 逆方向 → counter 倍率、NEUTRAL → 軽微縮小、それ以外 → 1.0。
    Returns: (htf_counter_trend, htf_lot_multiplier)
    """
    trend = str(htf_trend or "NEUTRAL").upper()
    counter = is_htf_trend_mismatch(trade_direction, trend)
    if counter:
        mult = counter_multiplier if counter_multiplier is not None else FVG_HTF_COUNTER_MULTIPLIER
        return True, float(mult)
    if trend == "NEUTRAL":
        mult = neutral_multiplier if neutral_multiplier is not None else FVG_HTF_NEUTRAL_MULTIPLIER
        return False, float(mult)
    return False, 1.0


def calc_fvg_candidate_score(
    setup: FvgFillSetup,
    gbp_setup: FvgFillSetup | None,
    eur_setup: FvgFillSetup | None,
    htf_aligned: bool,
) -> float:
    score = 0.0
    if gbp_setup and eur_setup:
        score += 15.0 if gbp_setup.direction == eur_setup.direction else 5.0
    score += min(25.0, setup.fvg_size_pips * 1.5)
    if setup.nearby_order_block_present:
        score += 20.0
    score += min(15.0, max(0.0, 20.0 - setup.nearby_order_block_distance_pips))
    if setup.volatility_20d > 0:
        vol_quality = min(1.0, setup.fvg_size_pips / max(setup.volatility_20d, 1.0))
        score += vol_quality * 15.0
    if htf_aligned:
        score += 10.0
    if setup.current_session in ("LONDON", "NY"):
        score += 10.0
    return round(max(0.0, min(100.0, score)), 2)


class FvgFillStrategy(BaseStrategy):
    """Strategy C (FVG Fill): FVG_LONG + FVG_SHORT 平均回帰 + Gemini L4 必須。"""

    def __init__(
        self,
        weights_config: dict[str, int] | None = None,
        mode_h1: bool = False,
        fvg_config: FvgFillConfig | None = None,
        bar_minutes: int = 15,
        enable_short: bool | None = None,
        enable_long: bool | None = None,
    ):
        super().__init__(weights_config, mode_h1)
        self.fvg_config = fvg_config or load_fvg_config()
        self.bar_minutes = bar_minutes
        self.enable_long = FVG_ENABLE_LONG if enable_long is None else enable_long
        self.enable_short = FVG_ENABLE_SHORT if enable_short is None else enable_short
        self._htf_gbp: pd.DataFrame | None = None
        self._htf_eur: pd.DataFrame | None = None
        # ① Bar-Lock Guard: pair -> floored bar timestamps audited
        self._audited_bars: dict[str, set[str]] = {}
        self._bar_audit_results: dict[str, dict[str, Any]] = {}

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def _bar_lock_key(self, pair: str, ts: pd.Timestamp) -> str:
        floored = pd.Timestamp(ts).floor(f"{self.bar_minutes}min")
        return f"{pair}|{floored.isoformat()}"

    def audit_with_bar_lock(
        self,
        pair: str,
        bar_ts: pd.Timestamp,
        audit_fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """同一執行足バー内の Gemini 重複呼び出しを物理遮断。"""
        key = self._bar_lock_key(pair, bar_ts)
        if key in self._bar_audit_results:
            return self._bar_audit_results[key]

        audited = self._audited_bars.setdefault(pair, set())
        if key in audited:
            return self._bar_audit_results.get(key, audit_fn())

        audited.add(key)
        result = audit_fn()
        self._bar_audit_results[key] = result
        return result

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[FvgFillSetup]:
        return detect_fvg_all_fill_setups(
            df,
            pair_name,
            h1_df,
            self.fvg_config,
            enable_long=self.enable_long,
            enable_short=self.enable_short,
        )

    def analyze_setup(
        self,
        setup: FvgFillSetup,
        gbp_setup: FvgFillSetup | None,
        eur_setup: FvgFillSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        h1_ref = h1_gbp if setup.pair in ("GBPUSD", "AUDUSD", "AUDJPY") else h1_eur
        htf_df = self._htf_gbp if setup.pair in ("GBPUSD", "AUDUSD", "AUDJPY") else self._htf_eur
        htf_result = analyze_htf_trend(h1_ref, setup.timestamp, htf_df=htf_df)
        htf_trend_direction = htf_result.direction
        htf_counter, htf_lot_multiplier = resolve_fvg_htf_lot_multiplier(
            setup.direction,
            htf_trend_direction,
        )
        htf_aligned = not htf_counter and htf_trend_direction != "NEUTRAL"

        smt_feats = calc_smt_features(gbp_setup, eur_setup, pip_size=pip_size_for_pair(setup.pair))
        candidate_score = calc_fvg_candidate_score(setup, gbp_setup, eur_setup, htf_aligned)
        atr_ratio = setup.sweep_distance / setup.atr if setup.atr > 0 else 0.0

        reason_codes: list[str] = []
        if htf_counter:
            reason_codes.append(CAUTION_HTF_COUNTER_TAG)
            reason_codes.append(HTF_TREND_MISMATCH_TAG)

        raw_features: dict[str, Any] = {
            "smt_intensity": smt_feats.intensity,
            "smt_diff": smt_feats.diff,
            "smt_leader": smt_feats.leader,
            "wick_ratio_pct": 0.0,
            "atr_ratio": round(atr_ratio, 4),
            "has_bos": False,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf_trend_direction,
            "htf_counter_trend": htf_counter,
            "htf_lot_multiplier": round(htf_lot_multiplier, 4),
            "fvg_top": setup.fvg_top,
            "fvg_bottom": setup.fvg_bottom,
            "fvg_size_pips": round(setup.fvg_size_pips, 2),
            "nearby_order_block_present": setup.nearby_order_block_present,
            "nearby_order_block_distance_pips": round(setup.nearby_order_block_distance_pips, 2),
            "volatility_20d": round(setup.volatility_20d, 4),
            "current_session": setup.current_session,
            "fvg_execute_min_confidence": FVG_EXECUTE_MIN_CONFIDENCE,
            "fvg_side": setup.fvg_side,
            "long_only": setup.direction == "BUY",
            "requires_fvg_gemini_audit": True,
            "l4_bypass": False,
            "htf_bypass": False,
            "htf_would_block": htf_counter,
            "htf_trend_hard_filter": False,
            "reject_reason": "",
            "reason_codes": reason_codes,
            "candidate_score": candidate_score,
        }

        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=candidate_score,
            raw_features=raw_features,
        )

    def evaluate(self, payload: dict, state: dict) -> StrategyResult:
        active: FvgFillSetup | None = state.get("active_setup")
        if active is None:
            return StrategyResult(is_setup=False, setup_type=self.setup_type, direction="")
        return self.analyze_setup(
            active,
            state.get("gbp_setup"),
            state.get("eur_setup"),
            state["h1_gbp"],
            state["h1_eur"],
        )
