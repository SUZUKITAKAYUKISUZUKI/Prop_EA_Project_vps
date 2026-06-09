"""
strategies/archive/asian_session_liquidity_sweep.py — Asian Session Liquidity Sweep (ALS)

ARCHIVED 2026-06: 取引回数が少ないため本番・標準 BT から外した。参照・再検討用。

平均回帰型（Mean Reversion）狙撃エンジン。
アジアレンジの流動性スイープ後、レンジ内への拒否（髭）と VWAP への引力を評価する。
対象: EURUSD / GBPUSD のみ。執行足 M5・構造/HTF M15（BT 既定）。L4 バイパス対応。L5 ピラミッド無効。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, time
from typing import Any, Callable, Literal

import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import analyze_htf_trend
from strategies.market_utils import (
    PIP_SIZE,
    SMTFeatures,
    calc_smt_features,
    compute_atr,
    pip_size_for_pair,
    positional_index as _positional_index,
    uses_primary_dataframe,
)

logger = logging.getLogger(__name__)

SETUP_TYPE = "ASIAN_SESSION_LIQUIDITY_SWEEP"
ALS_PAIR_PRIMARY = "GBPUSD"
ALS_PAIR_SECONDARY = "EURUSD"
ALLOWED_PAIRS = frozenset({ALS_PAIR_PRIMARY, ALS_PAIR_SECONDARY})

SweepDirection = Literal["BUY", "SELL"]
RejectReason = Literal[
    "REJECT_ALS_BREAKOUT_CLOSE",
    "REJECT_ALS_LOW_WICK",
    "REJECT_ALS_VWAP_NEAR",
    "REJECT_ALS_OVER_EXPANSION",
    "REJECT_ALS_UNDER_CONTRACTION",
    "REJECT_ALS_NO_INSIDE_RETURN",
    "REJECT_ALS_INVALID_TP",
]


@dataclass(frozen=True)
class AlsConfig:
    """ALS パラメータ — 環境変数 / .env から読み込み。"""

    asia_session_start: str = "00:00"
    asia_session_end: str = "06:00"
    london_trigger_start: str = "07:00"
    london_trigger_end: str = "11:00"
    min_atr_ratio: float = 0.4
    max_atr_ratio: float = 1.5
    sl_buffer_pips: float = 2.0
    min_wick_ratio_pct: float = 50.0
    ideal_wick_ratio_pct: float = 70.0
    min_vwap_deviation_ratio: float = 0.5
    ideal_vwap_deviation_ratio: float = 1.0
    min_rr_fallback: float = 1.5
    require_inside_return: bool = True
    l4_bypass: bool = True
    atr_period: int = 14


def load_als_config() -> AlsConfig:
    """`.env` / 環境変数から ALS 設定を構築。"""
    bypass_raw = os.getenv("ALS_L4_BYPASS", "1").strip().lower()
    l4_bypass = bypass_raw not in ("0", "false", "no", "off")
    inside_raw = os.getenv("ALS_REQUIRE_INSIDE_RETURN", "1").strip().lower()
    require_inside_return = inside_raw not in ("0", "false", "no", "off")
    return AlsConfig(
        asia_session_start=os.getenv("ASIA_SESSION_START", "00:00"),
        asia_session_end=os.getenv("ASIA_SESSION_END", "06:00"),
        london_trigger_start=os.getenv("LONDON_TRIGGER_START", "07:00"),
        london_trigger_end=os.getenv("LONDON_TRIGGER_END", "11:00"),
        min_atr_ratio=float(os.getenv("ALS_MIN_ATR_RATIO", "0.4")),
        max_atr_ratio=float(os.getenv("ALS_MAX_ATR_RATIO", "1.5")),
        sl_buffer_pips=float(os.getenv("ALS_SL_BUFFER_PIPS", "2.0")),
        min_wick_ratio_pct=float(os.getenv("ALS_MIN_WICK_RATIO_PCT", "50")),
        ideal_wick_ratio_pct=float(os.getenv("ALS_IDEAL_WICK_RATIO_PCT", "70")),
        min_vwap_deviation_ratio=float(os.getenv("ALS_MIN_VWAP_DEVIATION", "0.5")),
        ideal_vwap_deviation_ratio=float(os.getenv("ALS_IDEAL_VWAP_DEVIATION", "1.0")),
        min_rr_fallback=float(os.getenv("ALS_MIN_RR_FALLBACK", "1.5")),
        require_inside_return=require_inside_return,
        l4_bypass=l4_bypass,
        atr_period=int(os.getenv("ALS_ATR_PERIOD", "14")),
    )


def _parse_session_time(value: str) -> time:
    text = (value or "00:00").strip()
    parts = text.split(":")
    hour = int(parts[0]) if parts else 0
    minute = int(parts[1]) if len(parts) > 1 else 0
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return time(hour=hour, minute=minute)


def _hour_in_session(hour: int, start: time, end: time) -> bool:
    if start <= end:
        return start.hour <= hour <= end.hour
    return hour >= start.hour or hour <= end.hour


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if pd.isna(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _ensure_bars(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    if "hour" not in work.columns:
        work["hour"] = work["datetime"].dt.hour
    if "date" not in work.columns:
        work["date"] = work["datetime"].dt.date
    if "volume" not in work.columns:
        work["volume"] = 1.0
    else:
        work["volume"] = work["volume"].fillna(0.0).clip(lower=0.0)
    return work.sort_values("datetime").reset_index(drop=True)


def _normalize_pair(pair_name: str) -> str:
    upper = pair_name.upper().replace(".", "").replace("_", "").replace("-", "")
    if "GBPUSD" in upper:
        return "GBPUSD"
    if "EURUSD" in upper:
        return "EURUSD"
    if "AUDJPY" in upper:
        return "AUDJPY"
    if "USDJPY" in upper:
        return "USDJPY"
    if "AUDUSD" in upper:
        return "AUDUSD"
    if "NZDUSD" in upper:
        return "NZDUSD"
    return upper


def _compute_asia_range(
    df: pd.DataFrame,
    session_date: date,
    asia_start: time,
    asia_end: time,
    pip_size: float = PIP_SIZE,
) -> tuple[float, float, float] | None:
    day_mask = df["date"] == session_date
    hour_mask = df["hour"].apply(lambda h: _hour_in_session(int(h), asia_start, asia_end))
    session = df.loc[day_mask & hour_mask]
    if session.empty:
        return None
    try:
        asia_high = float(session["high"].max())
        asia_low = float(session["low"].min())
    except (TypeError, ValueError):
        return None
    if asia_high <= asia_low:
        return None
    asia_range_pips = (asia_high - asia_low) / pip_size
    return asia_high, asia_low, asia_range_pips


def _atr_at_bar(
    atr_series: pd.Series,
    structure_df: pd.DataFrame,
    bar_index: int,
    fallback_bar: pd.Series,
    pip_size: float = PIP_SIZE,
) -> float:
    try:
        if 0 <= bar_index < len(atr_series) and pd.notna(atr_series.iloc[bar_index]):
            return float(atr_series.iloc[bar_index])
    except (IndexError, TypeError, ValueError):
        pass
    try:
        return max(float(fallback_bar["high"] - fallback_bar["low"]), pip_size * 10)
    except (TypeError, ValueError):
        return pip_size * 10


def _evaluate_l2_liquidity_charge(
    asia_range_pips: float,
    h1_atr_price: float,
    config: AlsConfig,
    pip_size: float = PIP_SIZE,
) -> tuple[bool, RejectReason | None, float]:
    if h1_atr_price <= 0:
        return False, "REJECT_ALS_UNDER_CONTRACTION", 0.0
    h1_atr_pips = h1_atr_price / pip_size
    if h1_atr_pips <= 0:
        return False, "REJECT_ALS_UNDER_CONTRACTION", 0.0
    ratio = asia_range_pips / h1_atr_pips
    if ratio > config.max_atr_ratio:
        return False, "REJECT_ALS_OVER_EXPANSION", ratio
    if ratio < config.min_atr_ratio:
        return False, "REJECT_ALS_UNDER_CONTRACTION", ratio
    return True, None, ratio


def _bar_range(bar: pd.Series) -> float:
    return max(_safe_float(bar["high"]) - _safe_float(bar["low"]), PIP_SIZE)


def _close_inside_asia_range(close_p: float, asia_high: float, asia_low: float) -> bool:
    return asia_low <= close_p <= asia_high


def _is_breakout_close(
    close_p: float,
    asia_high: float,
    asia_low: float,
) -> bool:
    """終値がアジアレンジ外 → トレンドブレイクアウトとみなす。"""
    return close_p < asia_low or close_p > asia_high


def _compute_sweep_wick_ratio_pct(bar: pd.Series, direction: SweepDirection) -> float:
    """
    スイープ足の拒否髭比率（%）。

    BUY（安値スイープ）: 下髭 / レンジ
    SELL（高値スイープ）: 上髭 / レンジ
    """
    bar_range = _bar_range(bar)
    if bar_range <= 0:
        return 0.0
    open_p = _safe_float(bar["open"])
    close_p = _safe_float(bar["close"])
    high_p = _safe_float(bar["high"])
    low_p = _safe_float(bar["low"])
    if direction == "BUY":
        wick = min(open_p, close_p) - low_p
    else:
        wick = high_p - max(open_p, close_p)
    wick = max(wick, 0.0)
    return min(100.0, wick / bar_range * 100.0)


def _detect_sweep_direction(
    bar: pd.Series,
    asia_high: float,
    asia_low: float,
) -> SweepDirection | None:
    low_p = _safe_float(bar["low"])
    high_p = _safe_float(bar["high"])
    swept_low = low_p < asia_low
    swept_high = high_p > asia_high
    if swept_low and not swept_high:
        return "BUY"
    if swept_high and not swept_low:
        return "SELL"
    if swept_low and swept_high:
        # 両側スイープ — 終値に近い側の拒否を優先
        close_p = _safe_float(bar["close"])
        dist_low = abs(close_p - asia_low)
        dist_high = abs(close_p - asia_high)
        return "BUY" if dist_low <= dist_high else "SELL"
    return None


def _compute_vwap_to_index(df: pd.DataFrame, end_index: int) -> float | None:
    """
    session_date 先頭〜 end_index までの典型出来高加重 VWAP。
    volume 欠損時は等加重 close。
    """
    if end_index < 0 or end_index >= len(df):
        return None
    window = df.iloc[: end_index + 1]
    if window.empty:
        return None
    try:
        highs = window["high"].astype(float)
        lows = window["low"].astype(float)
        closes = window["close"].astype(float)
        typical = (highs + lows + closes) / 3.0
        volumes = window["volume"].astype(float).fillna(0.0)
        vol_sum = float(volumes.sum())
        if vol_sum <= 0:
            return float(closes.iloc[-1])
        return float((typical * volumes).sum() / vol_sum)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            return float(window["close"].iloc[-1])
        except (TypeError, ValueError, IndexError):
            return None


def _vwap_deviation_ratio(entry_price: float, vwap: float | None, h1_atr: float) -> float:
    if vwap is None or h1_atr <= 0:
        return 0.0
    return abs(entry_price - vwap) / h1_atr


def _compute_mean_reversion_tp(
    direction: SweepDirection,
    entry: float,
    stop_loss: float,
    asia_equilibrium: float,
    vwap: float | None,
    config: AlsConfig,
    pip_size: float = PIP_SIZE,
) -> tuple[float, str] | None:
    """
    平均回帰 TP: Equilibrium / VWAP を第一ターゲットとし、
    最低 RR を `min_rr_fallback * R` で確保する。
    """
    risk = abs(entry - stop_loss)
    if risk <= pip_size:
        return None
    min_distance = config.min_rr_fallback * risk

    if direction == "BUY":
        if entry >= asia_equilibrium - pip_size:
            return None
        magnets: list[tuple[str, float]] = [("EQUILIBRIUM", asia_equilibrium)]
        if vwap is not None and vwap > entry + pip_size:
            magnets.append(("VWAP", vwap))
        valid = [(label, price) for label, price in magnets if price > entry + pip_size]
        if not valid:
            return None
        tp_label, natural_tp = min(valid, key=lambda item: item[1])
        take_profit = max(natural_tp, entry + min_distance)
        if take_profit <= entry + pip_size:
            return None
        return take_profit, tp_label

    if entry <= asia_equilibrium + pip_size:
        return None
    magnets = [("EQUILIBRIUM", asia_equilibrium)]
    if vwap is not None and vwap < entry - pip_size:
        magnets.append(("VWAP", vwap))
    valid = [(label, price) for label, price in magnets if price < entry - pip_size]
    if not valid:
        return None
    tp_label, natural_tp = max(valid, key=lambda item: item[1])
    take_profit = min(natural_tp, entry - min_distance)
    if take_profit >= entry - pip_size:
        return None
    return take_profit, tp_label


@dataclass
class AlsSetup:
    """ALS 1 件の執行セットアップ（平均回帰型）。"""

    timestamp: pd.Timestamp
    pair: str
    direction: str
    asia_high: float
    asia_low: float
    asia_equilibrium_price: float
    asia_range_pips: float
    h1_atr: float
    asia_range_atr_ratio: float
    sweep_extreme: float
    sweep_bar_open: float
    sweep_bar_high: float
    sweep_bar_low: float
    sweep_bar_close: float
    wick_ratio_pct: float
    vwap: float
    vwap_deviation_ratio: float
    inside_return: bool
    tp_target_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    sweep_distance: float
    atr: float
    bar_index: int
    reason_codes: list[str] = field(default_factory=list)


def calc_als_candidate_score(
    setup: AlsSetup,
    gbp_setup: AlsSetup | None = None,
    eur_setup: AlsSetup | None = None,
) -> float:
    """
    平均回帰型 candidate_score（0–100）。

    LSFC 系の順張り評価軸は使用しない。
    """
    if setup.wick_ratio_pct < 50.0:
        return 0.0

    if setup.wick_ratio_pct >= 70.0:
        score = 70.0
    else:
        # 50–69%: 40–69 点へ線形補間
        span = max(70.0 - 50.0, 1.0)
        score = 40.0 + (setup.wick_ratio_pct - 50.0) * (30.0 / span)

    if setup.inside_return:
        score += 15.0

    if setup.vwap_deviation_ratio >= 1.0:
        score += 20.0
    elif setup.vwap_deviation_ratio >= 0.5:
        score += 10.0 * (setup.vwap_deviation_ratio - 0.5) / 0.5

    if gbp_setup and eur_setup and gbp_setup.direction == eur_setup.direction:
        score += 5.0

    return round(max(0.0, min(100.0, score)), 2)


def _build_als_setup(
    sweep_bar: pd.Series,
    entry_bar: pd.Series,
    bar_index: int,
    pair_name: str,
    direction: SweepDirection,
    asia_high: float,
    asia_low: float,
    asia_range_pips: float,
    h1_atr: float,
    asia_range_atr_ratio: float,
    wick_ratio_pct: float,
    vwap: float,
    vwap_deviation_ratio: float,
    inside_return: bool,
    config: AlsConfig,
    pip_size: float = PIP_SIZE,
) -> AlsSetup | None:
    try:
        asia_equilibrium = (asia_high + asia_low) / 2.0
        entry = _safe_float(entry_bar["close"])
        buffer = config.sl_buffer_pips * pip_size
        sweep_extreme = _safe_float(sweep_bar["low"]) if direction == "BUY" else _safe_float(sweep_bar["high"])
        if direction == "BUY":
            sweep_distance = max(asia_low - sweep_extreme, pip_size)
            stop_loss = sweep_extreme - buffer
        else:
            sweep_distance = max(sweep_extreme - asia_high, pip_size)
            stop_loss = sweep_extreme + buffer

        tp_result = _compute_mean_reversion_tp(
            direction,
            entry,
            stop_loss,
            asia_equilibrium,
            vwap,
            config,
            pip_size=pip_size,
        )
        if tp_result is None:
            return None
        take_profit, tp_target_type = tp_result

        reason_codes = [
            "ALS_MEAN_REVERSION",
            "ALS_SWEEP_REJECTED",
            f"ALS_WICK_{wick_ratio_pct:.0f}PCT",
            f"ALS_VWAP_DEV_{vwap_deviation_ratio:.2f}",
            f"ALS_TP_{tp_target_type}",
        ]
        if inside_return:
            reason_codes.append("ALS_INSIDE_RETURN")

        return AlsSetup(
            timestamp=pd.Timestamp(entry_bar["datetime"]),
            pair=pair_name,
            direction=direction,
            asia_high=asia_high,
            asia_low=asia_low,
            asia_equilibrium_price=round(asia_equilibrium, 6),
            asia_range_pips=round(asia_range_pips, 2),
            h1_atr=round(h1_atr, 6),
            asia_range_atr_ratio=round(asia_range_atr_ratio, 4),
            sweep_extreme=sweep_extreme,
            sweep_bar_open=_safe_float(sweep_bar["open"]),
            sweep_bar_high=_safe_float(sweep_bar["high"]),
            sweep_bar_low=_safe_float(sweep_bar["low"]),
            sweep_bar_close=_safe_float(sweep_bar["close"]),
            wick_ratio_pct=round(wick_ratio_pct, 2),
            vwap=round(vwap, 6),
            vwap_deviation_ratio=round(vwap_deviation_ratio, 4),
            inside_return=inside_return,
            tp_target_type=tp_target_type,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sweep_distance=sweep_distance,
            atr=h1_atr,
            bar_index=bar_index,
            reason_codes=reason_codes,
        )
    except (TypeError, ValueError, ZeroDivisionError):
        logger.debug("ALS setup build failed for %s", pair_name, exc_info=True)
        return None


def detect_asian_session_liquidity_sweep_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    config: AlsConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> list[AlsSetup]:
    """
    L1〜L3: アジアレンジ → L2 流動性 → ロンドン帯スイープ拒否 → インサイドリターン確認。
    """
    config = config or load_als_config()
    pair = _normalize_pair(pair_name)
    if pair not in ALLOWED_PAIRS:
        return []

    pip_size = pip_size_for_pair(pair)
    structure = _ensure_bars(h1_df if h1_df is not None else df)
    exec_df = _ensure_bars(df)
    if structure.empty or exec_df.empty:
        return []

    asia_start = _parse_session_time(config.asia_session_start)
    asia_end = _parse_session_time(config.asia_session_end)
    trigger_start = _parse_session_time(config.london_trigger_start)
    trigger_end = _parse_session_time(config.london_trigger_end)

    try:
        atr_series = compute_atr(structure, period=config.atr_period)
    except Exception:
        logger.exception("ALS: ATR calculation failed for %s", pair)
        return []

    setups: list[AlsSetup] = []
    emitted_dates: set[date] = set()
    scan_total = max(len(exec_df) - 2, 0)

    for i in range(1, len(exec_df) - 1):
        if progress_hook is not None:
            progress_hook(i, scan_total)
        sweep_bar = exec_df.iloc[i]
        entry_bar = exec_df.iloc[i + 1]
        try:
            ts = pd.Timestamp(entry_bar["datetime"])
            session_date = ts.date()
            entry_hour = int(entry_bar["hour"])
        except (TypeError, ValueError):
            continue

        if not _hour_in_session(entry_hour, trigger_start, trigger_end):
            continue

        try:
            from backtest_time_slice import is_in_als_bt_slice_window

            if not is_in_als_bt_slice_window(ts):
                continue
        except ImportError:
            pass

        if session_date in emitted_dates:
            continue

        asia_levels = _compute_asia_range(
            structure, session_date, asia_start, asia_end, pip_size=pip_size
        )
        if asia_levels is None:
            continue
        asia_high, asia_low, asia_range_pips = asia_levels

        struct_pos = int(
            structure["datetime"].searchsorted(pd.Timestamp(sweep_bar["datetime"]), side="left")
        )
        if struct_pos < len(structure) and pd.Timestamp(structure.iloc[struct_pos]["datetime"]) == pd.Timestamp(
            sweep_bar["datetime"]
        ):
            bar_pos = struct_pos
        else:
            bar_pos = min(i, len(structure) - 1)
        h1_atr = _atr_at_bar(atr_series, structure, bar_pos, sweep_bar, pip_size=pip_size)

        passed_l2, reject_reason, ratio = _evaluate_l2_liquidity_charge(
            asia_range_pips, h1_atr, config, pip_size=pip_size
        )
        if not passed_l2:
            logger.debug(
                "ALS L2 reject %s %s | ratio=%.2f reason=%s",
                pair,
                session_date,
                ratio,
                reject_reason,
            )
            continue

        sweep_dir = _detect_sweep_direction(sweep_bar, asia_high, asia_low)
        if sweep_dir is None:
            continue

        sweep_close = _safe_float(sweep_bar["close"])
        if _is_breakout_close(sweep_close, asia_high, asia_low):
            logger.debug(
                "ALS breakout reject %s %s | close=%.5f range=[%.5f, %.5f]",
                pair,
                session_date,
                sweep_close,
                asia_low,
                asia_high,
            )
            continue

        wick_ratio_pct = _compute_sweep_wick_ratio_pct(sweep_bar, sweep_dir)
        if wick_ratio_pct < config.min_wick_ratio_pct:
            logger.debug(
                "ALS low wick reject %s %s | wick_ratio=%.1f%%",
                pair,
                session_date,
                wick_ratio_pct,
            )
            continue

        entry_close = _safe_float(entry_bar["close"])
        inside_return = _close_inside_asia_range(entry_close, asia_high, asia_low)
        if config.require_inside_return and not inside_return:
            logger.debug(
                "ALS inside-return missing %s %s | entry_close=%.5f",
                pair,
                session_date,
                entry_close,
            )
            continue

        day_mask = structure["date"] == session_date
        day_df = structure.loc[day_mask].reset_index(drop=True)
        if day_df.empty:
            continue
        # 当日バー列内での entry 位置
        try:
            day_end = day_df.index[day_df["datetime"] == entry_bar["datetime"]]
            day_end_pos = int(day_end[0]) if len(day_end) > 0 else len(day_df) - 1
        except (TypeError, ValueError, IndexError):
            day_end_pos = len(day_df) - 1

        vwap = _compute_vwap_to_index(day_df, day_end_pos)
        if vwap is None:
            continue

        entry_price = entry_close
        if sweep_dir == "BUY" and entry_price >= (asia_high + asia_low) / 2.0 - pip_size:
            logger.debug(
                "ALS BUY entry above equilibrium %s %s | entry=%.5f eq=%.5f",
                pair,
                session_date,
                entry_price,
                (asia_high + asia_low) / 2.0,
            )
            continue
        if sweep_dir == "SELL" and entry_price <= (asia_high + asia_low) / 2.0 + pip_size:
            logger.debug(
                "ALS SELL entry below equilibrium %s %s | entry=%.5f eq=%.5f",
                pair,
                session_date,
                entry_price,
                (asia_high + asia_low) / 2.0,
            )
            continue

        vwap_dev = _vwap_deviation_ratio(entry_price, vwap, h1_atr)
        if vwap_dev < config.min_vwap_deviation_ratio:
            logger.debug(
                "ALS VWAP near reject %s %s | deviation=%.2f",
                pair,
                session_date,
                vwap_dev,
            )
            continue

        setup = _build_als_setup(
            sweep_bar,
            entry_bar,
            i + 1,
            pair,
            sweep_dir,
            asia_high,
            asia_low,
            asia_range_pips,
            h1_atr,
            ratio,
            wick_ratio_pct,
            vwap,
            vwap_dev,
            inside_return,
            config,
            pip_size=pip_size,
        )
        if setup is None:
            continue

        if calc_als_candidate_score(setup) <= 0.0:
            continue

        setups.append(setup)
        emitted_dates.add(session_date)

    if progress_hook is not None and scan_total > 0:
        progress_hook(scan_total, scan_total)

    return setups


class AsianSessionLiquiditySweepStrategy(BaseStrategy):
    """Asian Session Liquidity Sweep — 平均回帰型 BaseStrategy 実装。"""

    def __init__(
        self,
        weights_config: dict[str, int] | None = None,
        mode_h1: bool = False,
        als_config: AlsConfig | None = None,
    ):
        super().__init__(weights_config, mode_h1)
        self.als_config = als_config or load_als_config()
        self._htf_gbp: pd.DataFrame | None = None
        self._htf_eur: pd.DataFrame | None = None

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[AlsSetup]:
        return detect_asian_session_liquidity_sweep_setups(
            df, pair_name, h1_df, self.als_config
        )

    def detect_setup(self, market_data: dict[str, Any]) -> AlsSetup | None:
        pair = _normalize_pair(str(market_data.get("pair", "")))
        df = market_data.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        h1_df = market_data.get("h1_df")
        ts = market_data.get("bar_timestamp")
        setups = self.detect_setups(df, pair, h1_df if isinstance(h1_df, pd.DataFrame) else None)
        if not setups:
            return None
        if ts is not None:
            target = pd.Timestamp(ts)
            same = [s for s in setups if s.timestamp == target]
            if same:
                return same[0]
        return setups[-1]

    def analyze_setup(
        self,
        setup: AlsSetup,
        gbp_setup: AlsSetup | None,
        eur_setup: AlsSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        h1_ref = h1_gbp if uses_primary_dataframe(setup.pair) else h1_eur
        htf_df = self._htf_gbp if uses_primary_dataframe(setup.pair) else self._htf_eur
        htf_result = analyze_htf_trend(
            h1_ref,
            setup.timestamp,
            htf_df=htf_df,
        )
        htf_trend_direction = htf_result.direction

        smt_feats = calc_smt_features(gbp_setup, eur_setup)
        candidate_score = calc_als_candidate_score(setup, gbp_setup, eur_setup)
        l4_bypass = self.als_config.l4_bypass

        raw_features: dict[str, Any] = {
            "smt_intensity": smt_feats.intensity,
            "smt_diff": smt_feats.diff,
            "smt_leader": smt_feats.leader,
            "wick_ratio_pct": setup.wick_ratio_pct,
            "vwap_deviation_ratio": setup.vwap_deviation_ratio,
            "asia_equilibrium_price": setup.asia_equilibrium_price,
            "vwap": setup.vwap,
            "atr_ratio": round(
                setup.vwap_deviation_ratio if setup.h1_atr > 0 else 0.0,
                4,
            ),
            "has_bos": False,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf_trend_direction,
            "asia_high": setup.asia_high,
            "asia_low": setup.asia_low,
            "asia_range_pips": setup.asia_range_pips,
            "h1_atr": setup.h1_atr,
            "asia_range_atr_ratio": setup.asia_range_atr_ratio,
            "sweep_extreme": setup.sweep_extreme,
            "inside_return": setup.inside_return,
            "tp_target_type": setup.tp_target_type,
            "reversal_pattern": "MEAN_REVERSION",
            "reason_codes": list(setup.reason_codes),
            "htf_bypass": True,
            "htf_would_block": False,
            "l4_bypass": l4_bypass,
            "pyramid_layers": 0,
        }

        strategy_action = "ALLOW" if candidate_score > 0 else "REJECT"

        return StrategyResult(
            is_setup=candidate_score > 0,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=candidate_score,
            raw_features=raw_features,
            strategy_action=strategy_action,
        )

    def evaluate(self, market_data: dict[str, Any], account_state: dict[str, Any]) -> StrategyResult:
        active = market_data.get("active_setup") or account_state.get("active_setup")
        if active is None:
            detected = self.detect_setup(market_data)
            if detected is None:
                return StrategyResult(
                    is_setup=False,
                    setup_type=self.setup_type,
                    direction="FLAT",
                    strategy_action="REJECT",
                )
            active = detected

        gbp_s = market_data.get("gbp_setup") or account_state.get("gbp_setup")
        eur_s = market_data.get("eur_setup") or account_state.get("eur_setup")
        h1_gbp = market_data.get("h1_gbp") or account_state.get("h1_gbp")
        h1_eur = market_data.get("h1_eur") or account_state.get("h1_eur")
        if h1_gbp is None or h1_eur is None:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
                raw_features={"reject_reason": "missing_h1_data"},
            )
        return self.analyze_setup(active, gbp_s, eur_s, h1_gbp, h1_eur)


def als_pyramid_layers() -> int:
    """ALS は平均回帰のため L5 ピラミッドを常に無効。"""
    return 0


__all__ = [
    "ALS_PAIR_PRIMARY",
    "ALS_PAIR_SECONDARY",
    "ALS_SETUP_TYPE",
    "AlsConfig",
    "AlsSetup",
    "AsianSessionLiquiditySweepStrategy",
    "SETUP_TYPE",
    "als_pyramid_layers",
    "calc_als_candidate_score",
    "detect_asian_session_liquidity_sweep_setups",
    "load_als_config",
]

ALS_SETUP_TYPE = SETUP_TYPE
