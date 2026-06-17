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
DEFAULT_BT_LOT_SIZE = float(os.getenv("FINTOKEI_BT_DEFAULT_LOT", "0.01"))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("0", "false", "off", "no", "disabled"):
        return False
    if raw in ("1", "true", "yes", "on", "enabled"):
        return True
    return default


def is_bt_commission_enabled() -> bool:
    """Apply Fintokei round-trip commission in backtest / challenge simulation."""
    if _env_flag("BT_COMMISSION_ENABLED"):
        return True
    if _env_flag("FINTOKEI_BT_COMMISSION_ENABLED"):
        return True
    return _env_flag("FINTOKEI_BT_COMMISSION", True)


def round_trip_commission_usd(lot_size: float) -> float:
    if lot_size <= 0.0:
        return 0.0
    return FINTOKEI_COMMISSION_ROUND_TRIP_USD * lot_size


def resolve_bt_lot_size(
    lot_size: float | None,
    *,
    lot_factor: float = 1.0,
) -> float:
    if lot_size is not None and lot_size > 0.0:
        return float(lot_size)
    return max(DEFAULT_BT_LOT_SIZE, DEFAULT_BT_LOT_SIZE * max(float(lot_factor), 0.01))


def net_profit_r_after_commission(
    gross_profit_r: float,
    *,
    risk_budget_usd: float,
    lot_size: float | None = None,
    lot_factor: float = 1.0,
) -> tuple[float, float]:
    """Return (net_profit_r, commission_usd)."""
    if not is_bt_commission_enabled() or risk_budget_usd <= 0.0:
        return gross_profit_r, 0.0
    lot = resolve_bt_lot_size(lot_size, lot_factor=lot_factor)
    commission_usd = round_trip_commission_usd(lot)
    commission_r = commission_usd / risk_budget_usd
    return gross_profit_r - commission_r, commission_usd


def adjust_equity_for_executed_trade(
    equity: float,
    gross_profit_r: float,
    risk_budget_usd: float,
    *,
    lot_size: float | None = None,
    lot_factor: float = 1.0,
) -> tuple[float, float, float]:
    """Return (equity_after, net_profit_r, commission_usd)."""
    net_r, commission_usd = net_profit_r_after_commission(
        gross_profit_r,
        risk_budget_usd=risk_budget_usd,
        lot_size=lot_size,
        lot_factor=lot_factor,
    )
    return equity + risk_budget_usd * net_r, net_r, commission_usd


def finalize_portfolio_fintokei_commission(
    df,
    *,
    profile: str = "challenge",
    starting_equity: float | None = None,
):
    """
    Walk executed trades chronologically; convert gross R to net R after Fintokei commission.

    Expects gross_profit_r (falls back to profit_r). Updates profit_r, sized_result_r,
    equity_before_trade, equity_after_trade, commission_usd.
    """
    import pandas as pd

    from audit.risk_manager import STARTING_EQUITY, effective_base_risk_pct

    if df.empty or not is_bt_commission_enabled():
        return df

    work = df.sort_values("timestamp").copy().reset_index(drop=True)
    if "gross_profit_r" not in work.columns:
        work["gross_profit_r"] = pd.to_numeric(work.get("profit_r", 0.0), errors="coerce").fillna(0.0)
    else:
        gross_fallback = pd.to_numeric(work.get("profit_r", 0.0), errors="coerce").fillna(0.0)
        work["gross_profit_r"] = pd.to_numeric(work["gross_profit_r"], errors="coerce").fillna(gross_fallback)

    equity = float(starting_equity or STARTING_EQUITY)
    phase_start = equity

    eq_before: list[float] = []
    eq_after: list[float] = []
    net_r_list: list[float] = []
    comm_list: list[float] = []
    sized_r_list: list[float] = []

    for row in work.itertuples(index=False):
        trade_result = str(getattr(row, "trade_result", ""))
        if trade_result not in ("WIN", "LOSS"):
            eq_before.append(equity)
            eq_after.append(equity)
            net_r_list.append(float(getattr(row, "profit_r", 0.0) or 0.0))
            comm_list.append(float(getattr(row, "commission_usd", 0.0) or 0.0))
            sized = getattr(row, "sized_result_r", None)
            sized_r_list.append(float(sized if sized is not None else 0.0))
            continue

        lot_factor = float(getattr(row, "lot_factor", 1.0) or 1.0)
        gross_r = float(getattr(row, "gross_profit_r", 0.0) or 0.0)
        lot_size = float(getattr(row, "final_lot_size", 0.0) or 0.0)
        base_risk = effective_base_risk_pct(profile, phase_start, equity)
        risk_usd = equity * base_risk * lot_factor
        net_r, commission = net_profit_r_after_commission(
            gross_r,
            risk_budget_usd=risk_usd,
            lot_size=lot_size if lot_size > 0 else None,
            lot_factor=lot_factor,
        )
        eq_b = equity
        equity = equity + risk_usd * net_r
        eq_before.append(eq_b)
        eq_after.append(equity)
        net_r_list.append(net_r)
        comm_list.append(commission)
        sized_r_list.append(net_r * lot_factor)

    work["equity_before_trade"] = eq_before
    work["equity_after_trade"] = eq_after
    work["profit_r"] = net_r_list
    work["commission_usd"] = comm_list
    if "sized_result_r" in work.columns:
        work["sized_result_r"] = sized_r_list
    return work


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
