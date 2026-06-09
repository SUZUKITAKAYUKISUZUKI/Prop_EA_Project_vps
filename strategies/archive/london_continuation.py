"""
strategies/archive/london_continuation.py — London Continuation（ロンドン順張り継続）

ARCHIVED 2026-06: 本番パイプラインから除外（資金効率の観点）。参照・legacy 検証用。

アジアレンジのエネルギー蓄積 → ロンドンオープン後の BOS（実体確定）→
FVG リテスト（不均衡埋め）でモメンタムに追随する順張りロジック。

時間軸は FT6 サーバー時刻ベース。夏冬差は ASIA_SESSION_HOURS / ENTRY_WINDOW_HOURS
定数の差し替えで調整可能。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from audit.risk_manager import MAX_DAILY_EXPOSURE_LIMIT_PCT
from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.market_utils import PIP_SIZE, compute_atr, positional_index as _positional_index
from strategies.session_dst import DATA_DST_TYPE, shift_hour, shift_hour_range

SETUP_TYPE = "LONDON_CONTINUATION"

# --- セッション定義（サーバー時間。UTC+2/3 等は定数差替で対応） ---
ASIA_SESSION_HOUR_START = 0
ASIA_SESSION_HOUR_END = 7                 # 00:00–07:59  アジアレンジ形成
LONDON_BOS_MIN_HOUR = 8                     # 08:00〜 BOS 探索開始
ENTRY_WINDOW_HOUR_START = 8
ENTRY_WINDOW_HOUR_END = 11                  # 08:00–11:59  ICT Silver Bullet 適応帯
ASIA_SESSION_HOURS = range(ASIA_SESSION_HOUR_START, ASIA_SESSION_HOUR_END + 1)
ENTRY_WINDOW_HOURS = range(ENTRY_WINDOW_HOUR_START, ENTRY_WINDOW_HOUR_END + 1)


def _resolve_continuation_session_hours(
    session_date: date,
    dst_type: str = DATA_DST_TYPE,
) -> tuple[range, int, range]:
    """
    対象日のアジア / ロンドン BOS / エントリー窗口時刻を返す。

    GMT_FIXED かつ米国 DST 期間内の日のみ、セッションを 1 時間前倒し（-1h）する。
    通常: アジア 0–7 / BOS 8+ / 窗口 8–11 → DST シフト: 0–6 / 7+ / 7–10
    """
    asia_hours = shift_hour_range(
        session_date,
        ASIA_SESSION_HOUR_START,
        ASIA_SESSION_HOUR_END,
        dst_type,
    )
    bos_min_hour = shift_hour(session_date, LONDON_BOS_MIN_HOUR, dst_type)
    entry_hours = shift_hour_range(
        session_date,
        ENTRY_WINDOW_HOUR_START,
        ENTRY_WINDOW_HOUR_END,
        dst_type,
    )
    return asia_hours, bos_min_hour, entry_hours

# --- 固定 pip 帯（後方互換・スコア fallback 用。検知フィルタは ATR 動的比率へ移行） ---
ASIA_RANGE_MIN_PIPS = 25.0
ASIA_RANGE_MAX_PIPS = 50.0

DEFAULT_BASE_RISK_PCT = 0.010               # 1.0% — L4.5 へ引き渡す戦略要求リスク
MIN_RR = 2.0

SPREAD_CAUTION_PIPS = float(os.getenv("LC_SPREAD_CAUTION_PIPS", "2.0"))
SPREAD_REJECT_PIPS = float(os.getenv("LC_SPREAD_REJECT_PIPS", "3.0"))


@dataclass(frozen=True)
class DynamicMomentumFilterConfig:
    """
    動的モメンタムフィルター強度。

    クオンツ設計意図:
      - アジアレンジ幅を直近 H1 ATR に正規化し、固定 pip 閾値のレジーム依存性を除去。
      - BOS 確定足の実体を過去 ATR 平均と比較し、注文フロー失速（短実体ブレイク）を除外。
    """

    asia_range_atr_lookback: int = 24       # 参照 ATR: 直近 24 本 H1 ≒ 1 営業日
    asia_range_min_atr_ratio: float = 0.50  # 下限: ATR の 50% 未満 → 収縮フェイク
    asia_range_max_atr_ratio: float = 2.00  # 上限: ATR の 200% 超 → ショック後の過拡張
    bos_body_atr_lookback: int = 14         # BOS 実体判定用 ATR 平均ウィンドウ
    bos_body_min_atr_ratio: float = 0.55    # 実体 ≥ 0.55 × ATR(14) — 仕様 0.5〜0.8 の下限寄り


def load_momentum_filter_config() -> DynamicMomentumFilterConfig:
    """環境変数でチューニング可能。未設定時はクオンツ既定値。"""
    return DynamicMomentumFilterConfig(
        asia_range_atr_lookback=int(os.getenv("LC_ASIA_RANGE_ATR_LOOKBACK", "24")),
        asia_range_min_atr_ratio=float(os.getenv("LC_ASIA_RANGE_MIN_ATR_RATIO", "0.50")),
        asia_range_max_atr_ratio=float(os.getenv("LC_ASIA_RANGE_MAX_ATR_RATIO", "2.00")),
        bos_body_atr_lookback=int(os.getenv("LC_BOS_BODY_ATR_LOOKBACK", "14")),
        bos_body_min_atr_ratio=float(os.getenv("LC_BOS_BODY_MIN_ATR_RATIO", "0.55")),
    )

BosType = Literal["BOS_BULLISH", "BOS_BEARISH", "NONE"]


@dataclass
class ContinuationSetup:
    """London Continuation 1 件の執行セットアップ。"""

    timestamp: pd.Timestamp
    pair: str
    direction: str
    asia_high: float
    asia_low: float
    asia_range_pips: float
    bos_type: BosType
    bos_bar_index: int
    fvg_top: float
    fvg_bottom: float
    entry_price: float
    stop_loss: float
    take_profit: float
    spread_pips: float
    bar_index: int
    atr: float
    in_silver_window: bool
    # 動的モメンタムフィルター診断値（L2 スコア / 監査ログ用）
    h1_atr_ref_pips: float = 0.0
    asia_range_atr_ratio: float = 0.0
    bos_body_atr_ratio: float = 0.0


def _mean_atr_pips(
    atr_series: pd.Series,
    bar_index: int,
    lookback: int,
) -> float:
    """
    指定バーまでの直近 lookback 本で ATR 平均を pip 換算。

    24 本 ≒ 1 日分の H1 ボラを「当日の市場エネルギー」基準として使用。
    """
    if bar_index < 0 or atr_series.empty:
        return 0.0
    start = max(0, bar_index - lookback + 1)
    window = atr_series.iloc[start : bar_index + 1].dropna()
    if window.empty:
        return 0.0
    return float(window.mean()) / PIP_SIZE


def _passes_asia_range_dynamic_filter(
    asia_range_pips: float,
    h1_atr_ref_pips: float,
    config: DynamicMomentumFilterConfig,
) -> bool:
    """
    ① アジアレンジ幅の動的 ATR フィルター。

    - ratio < min: ボラ収縮 → 微ブレイクはノイズ（往復ビンタの温床）
    - ratio > max: 前日ボラでレンジ過拡張 → モメンタム枯渇・平均回帰リスク
    """
    if h1_atr_ref_pips <= 0.0:
        return False
    ratio = asia_range_pips / h1_atr_ref_pips
    return config.asia_range_min_atr_ratio <= ratio <= config.asia_range_max_atr_ratio


def _bos_candle_body_size(bar: pd.Series) -> float:
    """BOS 確定足の実体長（price 単位）。"""
    return abs(float(bar["close"]) - float(bar["open"]))


def _bos_candle_has_body_energy(
    bar: pd.Series,
    atr_series: pd.Series,
    bar_index: int,
    config: DynamicMomentumFilterConfig,
) -> tuple[bool, float]:
    """
    ② BOS 確定足の実体エネルギーフィルター。

    実体が過去 14 本平均 ATR の bos_body_min_atr_ratio 未満の場合、
    長ヒゲ・短実体の「失速ブレイク」とみなし BOS から除外する。
    大口フローが伴う本物のトレンドは実体が ATR に対して十分な長さを持つ。
    """
    avg_atr_pips = _mean_atr_pips(atr_series, bar_index, config.bos_body_atr_lookback)
    if avg_atr_pips <= 0.0:
        return False, 0.0
    body_pips = _bos_candle_body_size(bar) / PIP_SIZE
    body_ratio = body_pips / avg_atr_pips
    return body_ratio >= config.bos_body_min_atr_ratio, body_ratio


def _body_breaks_above(bar: pd.Series, level: float) -> bool:
    """
    強気 BOS: 実体全体がレベル上方 — ヒゲのみの Sweep（close>level だが open<level）は除外。

    数理: min(open, close) > level ⟺ 確定足の実体が完全に level より上。
    """
    return float(min(bar["open"], bar["close"])) > level


def _body_breaks_below(bar: pd.Series, level: float) -> bool:
    """弱気 BOS: max(open, close) < level — 実体が完全に level 下方。"""
    return float(max(bar["open"], bar["close"])) < level


def _detect_bos_on_bar(
    bar: pd.Series,
    asia_high: float,
    asia_low: float,
    atr_series: pd.Series | None = None,
    bar_index: int = -1,
    momentum_config: DynamicMomentumFilterConfig | None = None,
) -> tuple[BosType, float]:
    """
    実体ブレイク + ② 実体エネルギー検証。

    Returns:
        (bos_type, bos_body_atr_ratio) — エネルギー不足時は ("NONE", ratio)
    """
    detected: BosType = "NONE"
    if _body_breaks_above(bar, asia_high):
        detected = "BOS_BULLISH"
    elif _body_breaks_below(bar, asia_low):
        detected = "BOS_BEARISH"

    if detected == "NONE":
        return "NONE", 0.0

    if atr_series is not None and bar_index >= 0 and momentum_config is not None:
        has_energy, body_ratio = _bos_candle_has_body_energy(
            bar, atr_series, bar_index, momentum_config
        )
        if not has_energy:
            return "NONE", body_ratio
        return detected, body_ratio

    return detected, 0.0


def _compute_bullish_fvg(highs: np.ndarray, lows: np.ndarray, end_idx: int) -> tuple[float, float] | None:
    """
    強気 FVG（3 本足不均衡）:
      gap = [high[i-2], low[i]]  where low[i] > high[i-2]
    戻り値: (fvg_bottom, fvg_top) = (high[i-2], low[i])
    """
    if end_idx < 2:
        return None
    bottom = float(highs[end_idx - 2])
    top = float(lows[end_idx])
    if top > bottom:
        return bottom, top
    return None


def _compute_bearish_fvg(highs: np.ndarray, lows: np.ndarray, end_idx: int) -> tuple[float, float] | None:
    """
    弱気 FVG:
      low[i-2] > high[i] → ギャップ [high[i], low[i-2]]
    """
    if end_idx < 2:
        return None
    bottom = float(highs[end_idx])
    top = float(lows[end_idx - 2])
    if top > bottom:
        return bottom, top
    return None


def _price_retests_fvg(bar: pd.Series, direction: str, fvg_bottom: float, fvg_top: float) -> bool:
    """FVG ゾーンへのリテスト: バーのレンジがギャップと交差する。"""
    if direction == "BUY":
        return float(bar["low"]) <= fvg_top and float(bar["high"]) >= fvg_bottom
    return float(bar["high"]) >= fvg_bottom and float(bar["low"]) <= fvg_top


def _price_retests_breakline(bar: pd.Series, direction: str, asia_high: float, asia_low: float) -> bool:
    """50% Discount/Premium またはブレイクラインへの戻り。"""
    mid = (asia_high + asia_low) / 2.0
    if direction == "BUY":
        return float(bar["low"]) <= asia_high and float(bar["close"]) >= mid
    return float(bar["high"]) >= asia_low and float(bar["close"]) <= mid


def _build_continuation_setup(
    bar: pd.Series,
    bar_index: int,
    pair_name: str,
    direction: str,
    asia_high: float,
    asia_low: float,
    asia_range_pips: float,
    bos_type: BosType,
    bos_bar_index: int,
    fvg_bottom: float,
    fvg_top: float,
    atr_val: float,
    in_silver_window: bool,
    spread_pips: float = 0.0,
    h1_atr_ref_pips: float = 0.0,
    asia_range_atr_ratio: float = 0.0,
    bos_body_atr_ratio: float = 0.0,
) -> ContinuationSetup | None:
    entry = float(bar["close"])
    if direction == "BUY":
        stop_loss = min(fvg_bottom, asia_high) - PIP_SIZE * 2
        risk = entry - stop_loss
        if risk <= 0:
            return None
        take_profit = entry + MIN_RR * risk
    else:
        stop_loss = max(fvg_top, asia_low) + PIP_SIZE * 2
        risk = stop_loss - entry
        if risk <= 0:
            return None
        take_profit = entry - MIN_RR * risk

    return ContinuationSetup(
        timestamp=bar["datetime"],
        pair=pair_name,
        direction=direction,
        asia_high=asia_high,
        asia_low=asia_low,
        asia_range_pips=asia_range_pips,
        bos_type=bos_type,
        bos_bar_index=bos_bar_index,
        fvg_top=fvg_top,
        fvg_bottom=fvg_bottom,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        spread_pips=spread_pips,
        bar_index=bar_index,
        atr=atr_val,
        in_silver_window=in_silver_window,
        h1_atr_ref_pips=h1_atr_ref_pips,
        asia_range_atr_ratio=asia_range_atr_ratio,
        bos_body_atr_ratio=bos_body_atr_ratio,
    )


def detect_london_continuation_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    spread_pips: float = 0.0,
    momentum_config: DynamicMomentumFilterConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> list[ContinuationSetup]:
    """
    日次ループ: アジアレンジ → BOS → FVG → リテストで ContinuationSetup を生成。

    v3.4+: ① アジアレンジ動的 ATR フィルター、② BOS 実体エネルギーフィルター適用。
    """
    config = momentum_config or load_momentum_filter_config()
    structure_df = h1_df if h1_df is not None else df
    if structure_df.empty:
        return []

    atr_series = compute_atr(structure_df)
    work = df.copy()
    work["date"] = work["datetime"].dt.date
    work["hour"] = work["datetime"].dt.hour

    setups: list[ContinuationSetup] = []

    day_groups = list(work.groupby("date"))
    day_total = len(day_groups)
    for day_idx, (session_date, day_bars) in enumerate(day_groups, start=1):
        if progress_hook is not None:
            progress_hook(day_idx, day_total)
        asia_hours, bos_min_hour, entry_hours = _resolve_continuation_session_hours(
            session_date
        )
        asia_bars = day_bars[day_bars["hour"].isin(asia_hours)]
        if len(asia_bars) < 2:
            continue

        asia_high = float(asia_bars["high"].max())
        asia_low = float(asia_bars["low"].min())
        asia_range_pips = (asia_high - asia_low) / PIP_SIZE

        # アジア終了時点の structure_df インデックスで 24 本 ATR 基準を取得
        last_asia = asia_bars.iloc[-1]
        asia_ref_match = structure_df.index[structure_df["datetime"] == last_asia["datetime"]]
        asia_ref_index = (
            _positional_index(structure_df, asia_ref_match[0])
            if len(asia_ref_match) > 0
            else 0
        )
        h1_atr_ref_pips = _mean_atr_pips(
            atr_series, asia_ref_index, config.asia_range_atr_lookback
        )
        asia_range_atr_ratio = (
            asia_range_pips / h1_atr_ref_pips if h1_atr_ref_pips > 0 else 0.0
        )

        # ① 動的 ATR フィルター（固定 25–50 pip 帯を置換）
        if not _passes_asia_range_dynamic_filter(
            asia_range_pips, h1_atr_ref_pips, config
        ):
            continue

        session_bars = day_bars[day_bars["hour"] >= bos_min_hour].sort_values("datetime")
        if session_bars.empty:
            continue

        bos_type: BosType = "NONE"
        bos_bar_index = -1
        bos_body_atr_ratio = 0.0
        fvg_bottom = 0.0
        fvg_top = 0.0
        direction = ""

        highs = structure_df["high"].to_numpy()
        lows = structure_df["low"].to_numpy()

        for idx_label, bar in session_bars.iterrows():
            bar_hour = int(bar["hour"])
            match_idx = structure_df.index[structure_df["datetime"] == bar["datetime"]]
            bar_index = (
                _positional_index(structure_df, match_idx[0])
                if len(match_idx) > 0
                else _positional_index(structure_df, idx_label)
            )
            bar_index = min(max(bar_index, 0), len(structure_df) - 1)

            if bos_type == "NONE":
                detected, body_ratio = _detect_bos_on_bar(
                    bar,
                    asia_high,
                    asia_low,
                    atr_series,
                    bar_index,
                    config,
                )
                if detected == "NONE":
                    continue
                bos_type = detected
                bos_bar_index = bar_index
                bos_body_atr_ratio = body_ratio
                direction = "BUY" if bos_type == "BOS_BULLISH" else "SELL"

                if bos_type == "BOS_BULLISH":
                    fvg = _compute_bullish_fvg(highs, lows, bar_index)
                else:
                    fvg = _compute_bearish_fvg(highs, lows, bar_index)

                if fvg is None:
                    bos_type = "NONE"
                    bos_body_atr_ratio = 0.0
                    continue
                fvg_bottom, fvg_top = fvg
                continue

            if bar_index <= bos_bar_index:
                continue

            in_silver = bar_hour in entry_hours
            if not in_silver:
                continue

            fvg_hit = _price_retests_fvg(bar, direction, fvg_bottom, fvg_top)
            line_hit = _price_retests_breakline(bar, direction, asia_high, asia_low)
            if not (fvg_hit or line_hit):
                continue

            atr_val = (
                float(atr_series.iloc[bar_index])
                if bar_index < len(atr_series) and pd.notna(atr_series.iloc[bar_index])
                else asia_high - asia_low
            )
            setup = _build_continuation_setup(
                bar,
                bar_index,
                pair_name,
                direction,
                asia_high,
                asia_low,
                asia_range_pips,
                bos_type,
                bos_bar_index,
                fvg_bottom,
                fvg_top,
                atr_val,
                in_silver,
                spread_pips,
                h1_atr_ref_pips,
                asia_range_atr_ratio,
                bos_body_atr_ratio,
            )
            if setup is not None:
                setups.append(setup)
                break

    return setups


def calc_continuation_candidate_score(
    setup: ContinuationSetup,
    momentum_config: DynamicMomentumFilterConfig | None = None,
) -> float:
    """0–100 候補スコア（L2 閾値判定用）。動的 ATR 比率ベース。"""
    config = momentum_config or load_momentum_filter_config()
    score = 0.0

    # アジアレンジ品質: 許容帯 [min, max] の中央（最適エネルギー帯）に近いほど高得点
    if setup.asia_range_atr_ratio > 0.0:
        optimal_ratio = (
            config.asia_range_min_atr_ratio + config.asia_range_max_atr_ratio
        ) / 2.0
        half_span = (
            config.asia_range_max_atr_ratio - config.asia_range_min_atr_ratio
        ) / 2.0
        if half_span > 0:
            dist = abs(setup.asia_range_atr_ratio - optimal_ratio) / half_span
            range_quality = max(0.0, 1.0 - dist)
        else:
            range_quality = 1.0
        score += range_quality * 25.0
    else:
        optimal_mid = (ASIA_RANGE_MIN_PIPS + ASIA_RANGE_MAX_PIPS) / 2.0
        range_quality = 1.0 - abs(setup.asia_range_pips - optimal_mid) / optimal_mid
        score += max(0.0, min(25.0, range_quality * 25.0))

    # BOS + 実体エネルギー: 閾値超えの程度で加点（本物モメンタムほど高スコア）
    if setup.bos_type != "NONE":
        if setup.bos_body_atr_ratio > 0.0 and config.bos_body_min_atr_ratio > 0:
            energy_quality = min(
                1.5,
                setup.bos_body_atr_ratio / config.bos_body_min_atr_ratio,
            )
            score += min(25.0, energy_quality / 1.5 * 25.0)
        else:
            score += 25.0

    fvg_width_pips = (setup.fvg_top - setup.fvg_bottom) / PIP_SIZE
    score += min(20.0, max(5.0, fvg_width_pips * 2.0))

    if setup.in_silver_window:
        score += 15.0

    if setup.h1_atr_ref_pips > 0:
        atr_ratio = setup.asia_range_pips / setup.h1_atr_ref_pips
    else:
        atr_ratio = (setup.asia_range_pips * PIP_SIZE) / setup.atr if setup.atr > 0 else 0.0
    score += min(15.0, atr_ratio * 10.0)

    if setup.spread_pips >= SPREAD_CAUTION_PIPS:
        score -= 10.0
    if setup.spread_pips >= SPREAD_REJECT_PIPS:
        score -= 20.0

    return round(max(0.0, min(100.0, score)), 2)


def _strategy_action_from_spread(spread_pips: float) -> str:
    if spread_pips >= SPREAD_REJECT_PIPS:
        return "REJECT"
    if spread_pips >= SPREAD_CAUTION_PIPS:
        return "CAUTION"
    return "ALLOW"


class LondonContinuationStrategy(BaseStrategy):
    """
    戦略 London Continuation（廃止 — legacy BT のみ）。

    L4.5 へのロット計算は行わず、base_risk_pct / SL / TP のみ StrategyResult へ返す。
    """

    def __init__(
        self,
        weights_config: dict[str, int] | None = None,
        mode_h1: bool = False,
        base_risk_pct: float = DEFAULT_BASE_RISK_PCT,
        momentum_config: DynamicMomentumFilterConfig | None = None,
    ):
        super().__init__(weights_config, mode_h1)
        self.base_risk_pct = base_risk_pct
        self.momentum_config = momentum_config or load_momentum_filter_config()

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[ContinuationSetup]:
        spread = float(os.getenv("LC_DEFAULT_SPREAD_PIPS", "0.0"))
        return detect_london_continuation_setups(
            df, pair_name, h1_df, spread, self.momentum_config
        )

    def analyze_setup(
        self,
        setup: ContinuationSetup,
        gbp_setup: ContinuationSetup | None,
        eur_setup: ContinuationSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        candidate_score = calc_continuation_candidate_score(setup, self.momentum_config)
        action = _strategy_action_from_spread(setup.spread_pips)

        atr_ratio = (
            setup.asia_range_atr_ratio
            if setup.asia_range_atr_ratio > 0
            else (
                (setup.asia_range_pips * PIP_SIZE) / setup.atr if setup.atr > 0 else 0.0
            )
        )

        metrics = {
            "asia_range_pips": round(setup.asia_range_pips, 2),
            "asia_high": setup.asia_high,
            "asia_low": setup.asia_low,
            "setup_score": round(candidate_score / 100.0, 4),
            "bos_detected": setup.bos_type != "NONE",
            "bos_type": setup.bos_type,
            "fvg_top": setup.fvg_top,
            "fvg_bottom": setup.fvg_bottom,
            "in_silver_window": setup.in_silver_window,
            "spread_pips": setup.spread_pips,
            "h1_atr_ref_pips": round(setup.h1_atr_ref_pips, 2),
            "asia_range_atr_ratio": round(setup.asia_range_atr_ratio, 4),
            "bos_body_atr_ratio": round(setup.bos_body_atr_ratio, 4),
            "atr_ratio": round(atr_ratio, 4),
        }

        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=candidate_score,
            strategy_action=action,
            base_risk_pct=self.base_risk_pct,
            raw_features={
                **metrics,
                "metrics": metrics,
                "has_bos": setup.bos_type != "NONE",
                "smt_intensity": 0.0,
                "smt_diff": 0.0,
                "smt_leader": "NONE",
                "wick_ratio_pct": 0.0,
                "both_sweep": False,
            },
        )

    def evaluate(self, market_data: dict[str, Any], account_state: dict[str, Any]) -> StrategyResult:
        """
        market_data: ohlcv/df, spread_pips, active_setup, h1_gbp, h1_eur, gbp_setup, eur_setup
        account_state: daily_committed_risk_pct, profile, ...
        """
        spread_pips = float(market_data.get("spread_pips", 0.0))
        committed = float(
            account_state.get(
                "daily_committed_risk_pct",
                account_state.get("committed_risk_pct", 0.0),
            )
        )
        base_risk = float(market_data.get("base_risk_pct", self.base_risk_pct))

        if committed + base_risk > MAX_DAILY_EXPOSURE_LIMIT_PCT:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
                base_risk_pct=base_risk,
                raw_features={
                    "reject_reason": "daily_exposure_limit",
                    "daily_committed_risk_pct": committed,
                    "requested_base_risk_pct": base_risk,
                    "metrics": {"bos_detected": False, "setup_score": 0.0},
                },
            )

        if spread_pips >= SPREAD_REJECT_PIPS:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
                base_risk_pct=base_risk,
                raw_features={
                    "reject_reason": "spread_too_wide",
                    "spread_pips": spread_pips,
                    "metrics": {"bos_detected": False, "setup_score": 0.0, "spread_pips": spread_pips},
                },
            )

        active = market_data.get("active_setup") or account_state.get("active_setup")
        if active is None:
            pair = str(market_data.get("pair", "GBPUSD"))
            df = market_data.get("df") or market_data.get("ohlcv")
            h1_df = market_data.get("h1_df")
            if isinstance(df, pd.DataFrame) and not df.empty:
                detected = detect_london_continuation_setups(
                    df,
                    pair,
                    h1_df,
                    spread_pips=spread_pips,
                    momentum_config=self.momentum_config,
                )
                ts = market_data.get("bar_timestamp")
                if ts is not None and detected:
                    ts_norm = pd.Timestamp(ts)
                    same = [s for s in detected if s.timestamp == ts_norm]
                    active = same[0] if same else detected[-1]
                elif detected:
                    active = detected[-1]

        if active is None:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
                raw_features={"reject_reason": "no_setup", "metrics": {"bos_detected": False}},
            )

        if isinstance(active, ContinuationSetup):
            active.spread_pips = spread_pips

        enriched_market = {
            **market_data,
            "active_setup": active,
            "spread_pips": spread_pips,
        }
        return super().evaluate(enriched_market, account_state)
