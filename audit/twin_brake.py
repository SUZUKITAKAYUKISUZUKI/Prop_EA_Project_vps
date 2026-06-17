"""
audit/twin_brake.py — Proximity & Daily DD Twin Brake (Fintokei 失格回避)

総DD近接ブレーキ (Proximity) と日次DD残量ブレーキ (Daily Smooth + Hard Stop) を乗算し、
エントリー直前の lot 倍率を動的に縮小する。

    final_lot_multiplier = proximity_multiplier × daily_multiplier   # 0.0 .. 1.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from audit.risk_manager import (
    MAX_DAILY_DD_PCT,
    STARTING_EQUITY,
    apply_lot_factor_floor,
    lot_from_risk_budget,
)

# --- Fintokei 失格ライン（参照値） ---
FINTOKEI_TOTAL_DD_DISQUALIFY_PCT = 8.5
FINTOKEI_DAILY_DD_LIMIT_PCT = MAX_DAILY_DD_PCT  # 5.0

# --- Proximity Brake（初期資金からの総損失率・小数 0.045 = 4.5%） ---
PROXIMITY_WARNING_LOSS_FRAC = float(
    os.getenv("TWIN_BRAKE_PROXIMITY_WARN_FRAC", "0.045")
)
PROXIMITY_SURVIVAL_LOSS_FRAC = float(
    os.getenv("TWIN_BRAKE_PROXIMITY_SURVIVE_FRAC", "0.065")
)
PROXIMITY_EMERGENCY_LOSS_FRAC = float(
    os.getenv("TWIN_BRAKE_PROXIMITY_EMERGENCY_FRAC", "0.075")
)
PROXIMITY_WARNING_MULT = float(os.getenv("TWIN_BRAKE_PROXIMITY_WARN_MULT", "0.50"))
PROXIMITY_SURVIVAL_MULT = float(os.getenv("TWIN_BRAKE_PROXIMITY_SURVIVE_MULT", "0.25"))
PROXIMITY_EMERGENCY_MULT = float(os.getenv("TWIN_BRAKE_PROXIMITY_EMERGENCY_MULT", "0.10"))

# --- Daily Brake（日次DD残量 % — v4.4 最終仕様） ---
DAILY_HARD_STOP_REMAINING_PCT = float(
    os.getenv("TWIN_BRAKE_DAILY_HARD_STOP_REMAINING", "1.5")
)
DAILY_SMOOTH_REMAINING_THRESHOLD_PCT = float(
    os.getenv("TWIN_BRAKE_DAILY_REMAINING_THRESHOLD", "4.0")
)
DAILY_SMOOTH_MULT = float(os.getenv("TWIN_BRAKE_DAILY_MULT", "0.50"))

REASON_TWIN_BRAKE_PROXIMITY = "TWIN_BRAKE_PROXIMITY"
REASON_TWIN_BRAKE_DAILY = "TWIN_BRAKE_DAILY"
REASON_TWIN_BRAKE_DAILY_HARD_STOP = "TWIN_BRAKE_DAILY_HARD_STOP"
REASON_TWIN_BRAKE_ACTIVE = "TWIN_BRAKE_ACTIVE"

ProximityMode = Literal["normal", "warning", "survival", "emergency"]


@dataclass(frozen=True)
class TwinBrakeBreakdown:
    """Twin Brake 計算結果（境界値テスト・ログ用）。"""

    proximity_multiplier: float
    daily_multiplier: float
    final_lot_multiplier: float
    total_loss_pct: float
    proximity_mode: ProximityMode
    daily_dd_remaining_percent: float


def is_twin_brake_enabled() -> bool:
    raw = os.getenv("TWIN_BRAKE_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "disabled")


def total_loss_pct_from_initial(
    current_equity: float,
    initial_balance: float = STARTING_EQUITY,
) -> float:
    """初期資金からの総損失率 (%)。含み益時は 0% 扱い。"""
    if initial_balance <= 0.0:
        return 0.0
    return max(0.0, (initial_balance - current_equity) / initial_balance * 100.0)


def current_loss_fraction(
    current_equity: float,
    initial_balance: float = STARTING_EQUITY,
) -> float:
    """初期資金からの総損失率（小数: 0.075 = 7.5%）。"""
    return total_loss_pct_from_initial(current_equity, initial_balance) / 100.0


def calculate_proximity_lot_factor(current_loss_pct: float) -> float:
    """
    総DD近接ブレーキ lot 倍率。

    Parameters
    ----------
    current_loss_pct : 初期資金に対する損失率（小数。例: 0.075 = 7.5%）
    """
    if current_loss_pct >= PROXIMITY_EMERGENCY_LOSS_FRAC:
        return PROXIMITY_EMERGENCY_MULT  # 緊急防空頭巾モード
    if current_loss_pct >= PROXIMITY_SURVIVAL_LOSS_FRAC:
        return PROXIMITY_SURVIVAL_MULT  # 延命モード
    if current_loss_pct >= PROXIMITY_WARNING_LOSS_FRAC:
        return PROXIMITY_WARNING_MULT  # 警戒モード
    return 1.0


def calculate_daily_brake_factor(daily_dd_remaining_percent: float) -> float:
    """
    日次DD残量連動ブレーキ lot 倍率（v4.4 最終仕様）。

    - 残量 <= 1.5%  → 0.0（Daily Hard Stop / 当日 3.5% 損で完全拒否）
    - 残量 <= 4.0%  → 0.50（Daily Smooth Brake / 当日 1.0% 損で即半減）
    - それ以外      → 1.0（通常巡航）
    """
    if daily_dd_remaining_percent <= DAILY_HARD_STOP_REMAINING_PCT:
        return 0.0
    if daily_dd_remaining_percent <= DAILY_SMOOTH_REMAINING_THRESHOLD_PCT:
        return DAILY_SMOOTH_MULT
    return 1.0


def proximity_brake_multiplier(
    total_loss_pct: float,
) -> tuple[float, ProximityMode]:
    """総損失率 (%) 入力のラッパー — `calculate_proximity_lot_factor` へ委譲。"""
    loss_frac = total_loss_pct / 100.0
    mult = calculate_proximity_lot_factor(loss_frac)
    if loss_frac >= PROXIMITY_EMERGENCY_LOSS_FRAC:
        mode: ProximityMode = "emergency"
    elif loss_frac >= PROXIMITY_SURVIVAL_LOSS_FRAC:
        mode = "survival"
    elif loss_frac >= PROXIMITY_WARNING_LOSS_FRAC:
        mode = "warning"
    else:
        mode = "normal"
    return mult, mode


def daily_smooth_brake_multiplier(daily_dd_remaining_percent: float) -> float:
    """後方互換ラッパー — `calculate_daily_brake_factor` へ委譲。"""
    return calculate_daily_brake_factor(daily_dd_remaining_percent)


def compute_twin_brake_multiplier(
    current_equity: float,
    initial_balance: float = STARTING_EQUITY,
    daily_dd_remaining_percent: float = FINTOKEI_DAILY_DD_LIMIT_PCT,
) -> TwinBrakeBreakdown:
    """
    Proximity × Daily の最終 lot 倍率を算出（0.0 .. 1.0）。

    Parameters
    ----------
    current_equity : 現在口座純資産
    initial_balance : 初期資金（デフォルト $100,000）
    daily_dd_remaining_percent : 日次DD残量 %（5.0 → 0.0）
    """
    total_loss = total_loss_pct_from_initial(current_equity, initial_balance)
    prox_mult, mode = proximity_brake_multiplier(total_loss)
    daily_mult = calculate_daily_brake_factor(daily_dd_remaining_percent)
    final_mult = round(min(1.0, max(0.0, prox_mult * daily_mult)), 4)

    return TwinBrakeBreakdown(
        proximity_multiplier=prox_mult,
        daily_multiplier=daily_mult,
        final_lot_multiplier=final_mult,
        total_loss_pct=round(total_loss, 4),
        proximity_mode=mode,
        daily_dd_remaining_percent=daily_dd_remaining_percent,
    )


def twin_brake_reason_tags(breakdown: TwinBrakeBreakdown) -> list[str]:
    """L6 reason_codes 用タグ（アクティブ時のみ）。"""
    if breakdown.final_lot_multiplier >= 1.0:
        return []
    tags: list[str] = [REASON_TWIN_BRAKE_ACTIVE]
    if breakdown.proximity_multiplier < 1.0:
        tags.append(REASON_TWIN_BRAKE_PROXIMITY)
    if breakdown.daily_multiplier <= 0.0:
        tags.append(REASON_TWIN_BRAKE_DAILY_HARD_STOP)
    elif breakdown.daily_multiplier < 1.0:
        tags.append(REASON_TWIN_BRAKE_DAILY)
    return tags


def apply_twin_brake_to_lot_factor(
    lot_factor: float,
    current_equity: float,
    sl_distance: float,
    base_risk_pct: float,
    *,
    initial_balance: float = STARTING_EQUITY,
    daily_dd_remaining_percent: float = FINTOKEI_DAILY_DD_LIMIT_PCT,
    pair: str | None = None,
) -> tuple[float, float, float, TwinBrakeBreakdown]:
    """
    既存 lot_factor に Twin Brake 倍率を乗算し RiskBudget / LotSize を再計算。

    Daily Hard Stop (×0.0) 時は lot / risk を完全ゼロ（フロア非適用）。
    """
    breakdown = compute_twin_brake_multiplier(
        current_equity,
        initial_balance,
        daily_dd_remaining_percent,
    )
    if not is_twin_brake_enabled() or lot_factor <= 0.0:
        risk_budget = round(current_equity * base_risk_pct * lot_factor, 2)
        lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor, pair=pair)
        return lot_factor, risk_budget, lot_size, breakdown

    if breakdown.final_lot_multiplier >= 1.0:
        risk_budget = round(current_equity * base_risk_pct * lot_factor, 2)
        lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor, pair=pair)
        return lot_factor, risk_budget, lot_size, breakdown

    if breakdown.final_lot_multiplier <= 0.0:
        return 0.0, 0.0, 0.0, breakdown

    lot_factor = round(lot_factor * breakdown.final_lot_multiplier, 4)
    lot_factor = apply_lot_factor_floor(lot_factor)
    risk_budget = round(current_equity * base_risk_pct * lot_factor, 2)
    lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor, pair=pair)
    return lot_factor, risk_budget, lot_size, breakdown
