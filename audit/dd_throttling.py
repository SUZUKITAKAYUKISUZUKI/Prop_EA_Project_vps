"""
audit/dd_throttling.py — Dynamic DD Throttling（3段階ロット減速）

エントリー拒絶ではなく、ピーク資産からの Current_DD_% に応じて final lot を縮小する。
"""

from __future__ import annotations

import os
from typing import Any

REASON_DD_THROTTLING_HALF = "DD_THROTTLING_HALF"
REASON_DD_THROTTLING_QUARTER = "DD_THROTTLING_QUARTER"
REASON_RECOVERY_BOOST = "RECOVERY_BOOST"
REASON_DAILY_STOP = "DAILY_STOP"

DEFAULT_DD_THROTTLE_CAUTION_PCT = 2.5
DEFAULT_DD_THROTTLE_DEFENSE_PCT = 6.0
DEFAULT_DD_THROTTLE_HALF_MULT = 0.5
DEFAULT_DD_THROTTLE_QUARTER_MULT = 0.25
DEFAULT_DD_THROTTLE_QUARTER_CONSECUTIVE_LOSSES = 2
DEFAULT_RECOVERY_BOOST_MULT = 1.1
DEFAULT_RECOVERY_BOOST_MIN_WINS = 2
DEFAULT_DAILY_STOP_CONSECUTIVE_LOSSES = 2
DEFAULT_DAILY_STOP_MIN_DD_PCT = 2.0


def is_dd_throttling_enabled() -> bool:
    raw = os.getenv("DD_THROTTLING_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "disabled")


def dd_throttle_caution_pct() -> float:
    return float(os.getenv("DD_THROTTLE_CAUTION_PCT", str(DEFAULT_DD_THROTTLE_CAUTION_PCT)))


def dd_throttle_defense_pct() -> float:
    return float(os.getenv("DD_THROTTLE_DEFENSE_PCT", str(DEFAULT_DD_THROTTLE_DEFENSE_PCT)))


def dd_throttle_half_mult() -> float:
    return float(os.getenv("DD_THROTTLE_HALF_MULT", str(DEFAULT_DD_THROTTLE_HALF_MULT)))


def dd_throttle_quarter_mult() -> float:
    return float(os.getenv("DD_THROTTLE_QUARTER_MULT", str(DEFAULT_DD_THROTTLE_QUARTER_MULT)))


def dd_throttle_quarter_consecutive_losses() -> int:
    return max(
        1,
        int(
            os.getenv(
                "DD_THROTTLE_QUARTER_CONSECUTIVE_LOSSES",
                str(DEFAULT_DD_THROTTLE_QUARTER_CONSECUTIVE_LOSSES),
            )
        ),
    )


def is_recovery_boost_enabled() -> bool:
    raw = os.getenv("RECOVERY_BOOST_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "disabled")


def recovery_boost_mult() -> float:
    return max(1.0, float(os.getenv("RECOVERY_BOOST_MULT", str(DEFAULT_RECOVERY_BOOST_MULT))))


def recovery_boost_min_wins() -> int:
    return max(1, int(os.getenv("RECOVERY_BOOST_MIN_WINS", str(DEFAULT_RECOVERY_BOOST_MIN_WINS))))


def is_daily_stop_enabled() -> bool:
    raw = os.getenv("DAILY_STOP_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "disabled")


def daily_stop_consecutive_losses() -> int:
    return max(
        1,
        int(
            os.getenv(
                "DAILY_STOP_CONSECUTIVE_LOSSES",
                str(DEFAULT_DAILY_STOP_CONSECUTIVE_LOSSES),
            )
        ),
    )


def daily_stop_min_dd_pct() -> float:
    raw = os.getenv("DAILY_STOP_MIN_DD_PCT", str(DEFAULT_DAILY_STOP_MIN_DD_PCT))
    return float(raw)


def is_daily_stop_active(account: Any) -> bool:
    """
    同日 N 連敗後、その日の残りエントリーを強制停止。

    TREF は 1 日最大 2 セットアップのため daily>=2 単体では発火しにくい。
    当日 1 敗 + グローバル N 連敗 + DD caution 以上で 2 件目を遮断する。
    """
    if not is_daily_stop_enabled():
        return False
    threshold = daily_stop_consecutive_losses()
    daily_streak = int(getattr(account, "daily_consecutive_losses", 0) or 0)
    global_streak = int(getattr(account, "consecutive_losses", 0) or 0)
    if daily_streak >= threshold:
        return True
    dd_pct = account.current_drawdown_pct()
    if (
        daily_streak >= 1
        and global_streak >= threshold
        and dd_pct >= daily_stop_min_dd_pct()
    ):
        return True
    return False


def register_executed_streak(account: Any, won: bool) -> None:
    """実執行 WIN/LOSS 後: 連勝・連敗とリカバリーブースト武装状態を更新。"""
    if won:
        account.consecutive_losses = 0
        account.consecutive_wins += 1
        account.daily_consecutive_losses = 0
        if account.consecutive_wins >= recovery_boost_min_wins():
            account.recovery_boost_armed = True
    else:
        account.consecutive_losses += 1
        account.consecutive_wins = 0
        account.recovery_boost_armed = False
        account.daily_consecutive_losses = int(
            getattr(account, "daily_consecutive_losses", 0) or 0
        ) + 1


def resolve_dd_throttle_tier(
    drawdown_pct: float,
    consecutive_losses: int = 0,
) -> tuple[float, str | None]:
    """
    Current_DD_% と連敗数からロット乗数と reason tag を返す。

    QUARTER（×0.25）:
      - consecutive_losses >= 2（2連敗直後）
      - Current_DD_% >= 6.0%
    HALF（×0.5）:
      - Current_DD_% >= 2.5%
    通常:
      - 上記以外
    """
    if consecutive_losses >= dd_throttle_quarter_consecutive_losses():
        return dd_throttle_quarter_mult(), REASON_DD_THROTTLING_QUARTER
    if drawdown_pct >= dd_throttle_defense_pct():
        return dd_throttle_quarter_mult(), REASON_DD_THROTTLING_QUARTER
    if drawdown_pct >= dd_throttle_caution_pct():
        return dd_throttle_half_mult(), REASON_DD_THROTTLING_HALF
    return 1.0, None


def apply_dynamic_dd_throttling(
    lot_factor: float,
    equity: float,
    sl_distance: float,
    base_risk_pct: float,
    drawdown_pct: float,
    consecutive_losses: int = 0,
) -> tuple[float, float, float, float, str | None]:
    """Position Sizing Layer: DD tier + 連敗に応じて lot_factor / lot_size を縮小。"""
    from audit.risk_manager import apply_lot_factor_floor, lot_from_risk_budget

    throttle_mult, tag = resolve_dd_throttle_tier(drawdown_pct, consecutive_losses)
    if throttle_mult >= 1.0 or lot_factor <= 0.0:
        risk_budget = round(equity * base_risk_pct * lot_factor, 2)
        lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
        return lot_factor, risk_budget, lot_size, throttle_mult, None

    lot_factor = round(lot_factor * throttle_mult, 4)
    lot_factor = apply_lot_factor_floor(lot_factor)
    risk_budget = round(equity * base_risk_pct * lot_factor, 2)
    lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
    return lot_factor, risk_budget, lot_size, throttle_mult, tag


def apply_recovery_boost(
    lot_factor: float,
    equity: float,
    sl_distance: float,
    base_risk_pct: float,
    recovery_boost_armed: bool,
) -> tuple[float, float, float, float, str | None]:
    """
    連勝後の次トレード1回のみ lot をブースト（V字回復加速）。

    武装済みフラグを消費するのは呼び出し側（main_platform）。
    """
    from audit.risk_manager import apply_lot_factor_floor, lot_from_risk_budget

    if not is_recovery_boost_enabled() or not recovery_boost_armed or lot_factor <= 0.0:
        risk_budget = round(equity * base_risk_pct * lot_factor, 2)
        lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
        return lot_factor, risk_budget, lot_size, 1.0, None

    boost_mult = recovery_boost_mult()
    lot_factor = round(lot_factor * boost_mult, 4)
    lot_factor = apply_lot_factor_floor(lot_factor)
    risk_budget = round(equity * base_risk_pct * lot_factor, 2)
    lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
    return lot_factor, risk_budget, lot_size, boost_mult, REASON_RECOVERY_BOOST
