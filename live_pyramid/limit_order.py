"""Convert PyramidManager add intent to Limit-order parameters (BT close parity)."""

from __future__ import annotations

from dataclasses import dataclass

from pyramid_manager import PyramidManager


@dataclass(frozen=True)
class PyramidLimitIntent:
    limit_price: float
    lot_size: float
    unified_sl: float
    layer_index: int
    ttl_bars: int


def preview_pyramid_ratchet_sl(mgr: PyramidManager, price: float) -> float:
    """add_pyramid_layer() 実行前の unified SL（ratchet 後）を非破壊的に算出。"""
    if mgr.direction == "BUY":
        ratchet = price - 0.5 * mgr.atr
        return max(mgr.unified_stop_loss(), ratchet)
    ratchet = price + 0.5 * mgr.atr
    return min(mgr.unified_stop_loss(), ratchet)


def build_pyramid_limit_intent(
    mgr: PyramidManager,
    close: float,
    *,
    ttl_bars: int,
) -> PyramidLimitIntent:
    """
    L5 BT の add_pyramid_layer(close) と同一参照価格・lot・SL を Limit 指値へ変換。

    BT は bar.close で即時約定。ライブは limit_price=close の Buy/Sell Limit を発注する。
    """
    return PyramidLimitIntent(
        limit_price=round(close, 5),
        lot_size=round(mgr.pyramid_lot_for_next_layer(), 4),
        unified_sl=round(preview_pyramid_ratchet_sl(mgr, close), 5),
        layer_index=mgr.layer_count + 1,
        ttl_bars=max(1, int(ttl_bars)),
    )
