"""
volume_profile_analyzer.py — セッション累積出来高から VAH / VAL / POC を算出

Usage:
    from volume_profile_analyzer import SessionVolumeProfile

    profile = SessionVolumeProfile.for_pair("GBPUSD")
    levels = profile.calculate_profile(session_m5_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

VALUE_AREA_FRACTION = 0.70
DEFAULT_VP_BUFFER_PIPS = 2.0


@dataclass(frozen=True)
class VpLocationScoreTiers:
    """VP ロケーションスコア段階（L2 / Bayes 用。env や config で上書き可）。"""

    sweep_zone: int = 30
    favorable: int = 10
    neutral: int = 0
    adverse: int = -20


DEFAULT_VP_LOCATION_SCORE_TIERS = VpLocationScoreTiers()

TradeDirection = Literal["BUY", "SELL"]


def normalize_trade_direction(direction: str | None) -> TradeDirection:
    """BUY/SELL/LONG/SHORT 表記揺れを VP-VAR 用に正規化。"""
    side = str(direction or "BUY").strip().upper()
    if side in ("LONG", "BUY"):
        return "BUY"
    if side in ("SHORT", "SELL"):
        return "SELL"
    return "BUY"


class VolumeProfileLevels(TypedDict):
    vah: float
    val: float
    poc: float


class SessionVolumeProfile:
    """セッションごとの累積出来高から VAH/VAL/POCをリアルタイム算出するクオンツモジュール"""

    def __init__(self, price_tick_size: float = 0.00001, bin_step_pips: float = 0.5):
        self.tick_size = price_tick_size
        self.bin_step = bin_step_pips * (10.0 * price_tick_size)  # 0.5pips刻みで価格帯をグループ化

    @classmethod
    def for_pair(cls, pair: str, bin_step_pips: float = 0.5) -> SessionVolumeProfile:
        """pip サイズから tick / bin 幅を自動設定（JPY クロス対応）。"""
        from strategies.market_utils import pip_size_for_pair

        pip = pip_size_for_pair(pair)
        return cls(price_tick_size=pip / 10.0, bin_step_pips=bin_step_pips)

    def calculate_profile(self, session_df: pd.DataFrame) -> VolumeProfileLevels:
        """session_df: セッション開始（例:ロンドン16時）から現在までのM5/M1データ

        返り値: {'vah': float, 'val': float, 'poc': float}
        """
        empty: VolumeProfileLevels = {"vah": np.nan, "val": np.nan, "poc": np.nan}
        if session_df.empty:
            return empty

        # 破壊的変更（副作用）を防ぐため、元 DataFrame とは独立したコピーで処理
        df = session_df.copy()
        if "volume" not in df.columns:
            df["volume"] = 1.0
        else:
            df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)

        df["bin"] = (df["close"] / self.bin_step).round() * self.bin_step
        profile = df.groupby("bin", sort=True)["volume"].sum()

        if profile.empty or float(profile.sum()) <= 0.0:
            return empty

        # 2. POCの特定（最も出来高が多いBin）
        poc = float(profile.idxmax())

        # 3. 出来高の70%が収まるバリューエリア（VAH / VAL）の計算
        total_volume = float(profile.sum())
        target_volume = total_volume * VALUE_AREA_FRACTION

        bins = profile.index.tolist()
        volumes = profile.values.tolist()
        poc_idx = bins.index(poc)

        lower_idx = poc_idx
        upper_idx = poc_idx
        current_volume = float(volumes[poc_idx])

        while current_volume < target_volume:
            has_lower = lower_idx > 0
            has_upper = upper_idx < len(bins) - 1

            if not has_lower and not has_upper:
                break

            v_lower = float(volumes[lower_idx - 1]) if has_lower else -1.0
            v_upper = float(volumes[upper_idx + 1]) if has_upper else -1.0

            if v_lower >= v_upper:
                lower_idx -= 1
                current_volume += v_lower
            else:
                upper_idx += 1
                current_volume += v_upper

        return {"vah": float(bins[upper_idx]), "val": float(bins[lower_idx]), "poc": poc}

    def evaluate_vp_location(
        self,
        direction: TradeDirection,
        profile: VolumeProfileLevels,
        *,
        pip_size: float,
        filter_price: float,
        score_price: float | None = None,
        buffer_pips: float | None = None,
        buffer_atr: float | None = None,
        score_tiers: VpLocationScoreTiers | None = None,
    ) -> tuple[bool, int]:
        """VP-VAR: (エントリー許可, ロケーションスコア) を返す。

        filter_price: SWEEP 等のトリガー判定用（BUY=スウィープ安値, SELL=スウィープ高値）
        score_price: L2/Bayes 用（省略時は filter_price）
        buffer: buffer_atr が正なら ATR ベース、そうでなければ buffer_pips * pip_size
        """
        direction = normalize_trade_direction(direction)
        val = profile.get("val")
        vah = profile.get("vah")
        poc = profile.get("poc")
        if val is None or vah is None or poc is None or np.isnan(val) or np.isnan(vah) or np.isnan(poc):
            return False, 0

        tiers = score_tiers or DEFAULT_VP_LOCATION_SCORE_TIERS
        if buffer_atr is not None and buffer_atr > 0:
            buffer = float(buffer_atr)
        else:
            pips = DEFAULT_VP_BUFFER_PIPS if buffer_pips is None else buffer_pips
            buffer = pips * pip_size

        price_for_score = filter_price if score_price is None else score_price
        is_allowed = False
        location_score = tiers.neutral

        if direction == "BUY":
            if filter_price <= val + buffer:
                is_allowed = True
            if price_for_score <= val:
                location_score = tiers.sweep_zone
            elif val < price_for_score <= poc:
                location_score = tiers.favorable
            elif poc < price_for_score <= vah:
                location_score = tiers.neutral
            else:
                location_score = tiers.adverse
        elif direction == "SELL":
            if filter_price >= vah - buffer:
                is_allowed = True
            if price_for_score >= vah:
                location_score = tiers.sweep_zone
            elif vah > price_for_score >= poc:
                location_score = tiers.favorable
            elif poc > price_for_score >= val:
                location_score = tiers.neutral
            else:
                location_score = tiers.adverse

        return is_allowed, int(location_score)


__all__ = [
    "DEFAULT_VP_BUFFER_PIPS",
    "DEFAULT_VP_LOCATION_SCORE_TIERS",
    "SessionVolumeProfile",
    "VALUE_AREA_FRACTION",
    "VolumeProfileLevels",
    "VpLocationScoreTiers",
    "normalize_trade_direction",
]
