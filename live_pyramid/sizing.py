"""Live pyramid lot caps — align with Fintokei single-position rule."""

from __future__ import annotations

from strategies.market_utils import normalize_pair_name


def cap_pyramid_lot_size(
    lot_size: float,
    *,
    symbol: str,
    sl_price: float,
    reference_price: float,
    equity: float,
    phase_start_equity: float,
) -> float:
    """Cap pyramid layer lot so 1R loss at unified SL stays within prop limits."""
    if lot_size <= 0.0 or equity <= 0.0:
        return lot_size
    pair = normalize_pair_name(symbol) or symbol.upper()
    sl_distance = abs(reference_price - sl_price)
    if sl_distance <= 0.0:
        return lot_size
    from audit.risk_manager import cap_lot_size_to_max_loss_pct

    ref = phase_start_equity if phase_start_equity > 0 else equity
    capped, _ = cap_lot_size_to_max_loss_pct(
        lot_size,
        pair=pair,
        sl_distance=sl_distance,
        equity=equity,
        reference_equity=ref,
    )
    return capped
