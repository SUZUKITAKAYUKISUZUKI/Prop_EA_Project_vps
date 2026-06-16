"""
Fintokei 等ブローカー手数料を SL / トレーリングに反映するユーティリティ。

既定: 片道 USD 3 / 1.0 lot → 往復 USD 6 / 1.0 lot。
SL 決済時のネット損益が手数料 + 微益バッファ以上になるよう価格下限を算出する。
"""

from __future__ import annotations

import os
from typing import Sequence

FINTOKEI_COMMISSION_ONE_WAY_USD = float(os.getenv("FINTOKEI_COMMISSION_ONE_WAY_USD", "3"))
FINTOKEI_COMMISSION_ROUND_TRIP_USD = float(
    os.getenv(
        "FINTOKEI_COMMISSION_ROUND_TRIP_USD",
        str(FINTOKEI_COMMISSION_ONE_WAY_USD * 2),
    )
)
FINTOKEI_MIN_NET_PROFIT_USD = float(os.getenv("FINTOKEI_MIN_NET_PROFIT_USD", "0.50"))

# BT フォールバック（MT5 tick 未指定時）— 主要 FX 1 pip ≈ $10/lot
DEFAULT_FOREX_TICK_SIZE = 0.0001
DEFAULT_FOREX_TICK_VALUE_PER_LOT = 10.0
DEFAULT_JPY_TICK_SIZE = 0.01
DEFAULT_JPY_TICK_VALUE_PER_LOT = 10.0
DEFAULT_XAU_TICK_SIZE = 0.01
DEFAULT_XAU_TICK_VALUE_PER_LOT = 1.0


def resolve_tick_metrics(
    tick_size: float,
    tick_value: float,
    symbol: str = "",
) -> tuple[float, float]:
    """Return (tick_size, tick_value per 1.0 lot) for commission → price conversion."""
    if tick_size > 0 and tick_value > 0:
        return tick_size, tick_value
    sym = str(symbol).upper()
    if "XAU" in sym or "GOLD" in sym:
        return DEFAULT_XAU_TICK_SIZE, DEFAULT_XAU_TICK_VALUE_PER_LOT
    if "JPY" in sym:
        return DEFAULT_JPY_TICK_SIZE, DEFAULT_JPY_TICK_VALUE_PER_LOT
    return DEFAULT_FOREX_TICK_SIZE, DEFAULT_FOREX_TICK_VALUE_PER_LOT


def target_gross_usd_at_exit(
    total_lot: float,
    *,
    commission_round_trip_per_lot: float | None = None,
    min_net_profit_usd: float | None = None,
) -> float:
    if total_lot <= 0:
        return 0.0
    comm = (
        commission_round_trip_per_lot
        if commission_round_trip_per_lot is not None
        else FINTOKEI_COMMISSION_ROUND_TRIP_USD
    )
    min_net = min_net_profit_usd if min_net_profit_usd is not None else FINTOKEI_MIN_NET_PROFIT_USD
    return comm * total_lot + min_net


def min_net_profit_sl(
    direction: str,
    legs: Sequence[tuple[float, float]],
    *,
    tick_size: float = 0.0,
    tick_value: float = 0.0,
    symbol: str = "",
    commission_round_trip_per_lot: float | None = None,
    min_net_profit_usd: float | None = None,
) -> float:
    """
    全 leg 同一 SL 決済時に手数料込みで微益以上となる SL 価格下限/上限。

    legs: (entry_price, lot_size) の列。
    """
    if not legs:
        return 0.0
    total_lot = sum(float(lot) for _, lot in legs)
    if total_lot <= 0:
        return float(legs[0][0])

    weighted_entry = sum(float(entry) * float(lot) for entry, lot in legs) / total_lot
    ts, tv = resolve_tick_metrics(tick_size, tick_value, symbol)
    target_usd = target_gross_usd_at_exit(
        total_lot,
        commission_round_trip_per_lot=commission_round_trip_per_lot,
        min_net_profit_usd=min_net_profit_usd,
    )
    price_lot_sum = target_usd * ts / tv
    delta = price_lot_sum / total_lot
    side = str(direction).upper()
    if side == "BUY":
        return weighted_entry + delta
    if side == "SELL":
        return weighted_entry - delta
    return weighted_entry


def apply_commission_sl_floor(direction: str, sl: float, floor_sl: float) -> float:
    """Ratchet 後 SL が手数料込み微益下限を割らないようクランプ。"""
    side = str(direction).upper()
    if side == "BUY":
        return max(sl, floor_sl)
    if side == "SELL":
        return min(sl, floor_sl)
    return sl
