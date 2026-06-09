"""CSPA live exit management — shared config and bar-step tracker for EA / mt5_executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from strategies.cspa import (
    CSPA_BE_ARM_MFE_R,
    CSPA_BE_BUFFER_ATR,
    CSPA_BE_ENABLED,
    CSPA_BE_PULLBACK_CLOSE_R,
    CSPA_BE_RHYTHM_MAX_BARS,
    CSPA_BE_TRIGGER_MFE_R,
    CSPA_TRAIL_ATR_MULT,
    CSPA_TRAIL_ENABLED,
    CspaSetup,
    SETUP_TYPE as CSPA_SETUP_TYPE,
    TradeDirection,
    is_cspa_be_trail_enabled,
)

TradeDirectionStr = Literal["BUY", "SELL"]


def build_cspa_exit_signal_fields(setup: CspaSetup) -> dict[str, Any]:
    """MT5 EA / bridge 向け CSPA 出口管理パラメータ（フラット JSON キー）。"""
    if not is_cspa_be_trail_enabled():
        return {"setup_type": CSPA_SETUP_TYPE, "exit_mode": "FIXED_SL"}

    return {
        "setup_type": CSPA_SETUP_TYPE,
        "exit_mode": "CSPA_BE_TRAIL",
        "exit_atr": round(float(setup.momentum.atr), 6),
        "exit_be_enabled": int(CSPA_BE_ENABLED),
        "exit_trail_enabled": int(CSPA_TRAIL_ENABLED),
        "exit_be_arm_mfe_r": CSPA_BE_ARM_MFE_R,
        "exit_be_trigger_mfe_r": CSPA_BE_TRIGGER_MFE_R,
        "exit_be_pullback_close_r": CSPA_BE_PULLBACK_CLOSE_R,
        "exit_be_rhythm_max_bars": CSPA_BE_RHYTHM_MAX_BARS,
        "exit_trail_atr_mult": CSPA_TRAIL_ATR_MULT,
        "exit_be_buffer_atr": CSPA_BE_BUFFER_ATR,
    }


def _profit_r(direction: TradeDirection, entry: float, price: float, initial_risk: float) -> float:
    if initial_risk <= 0:
        return 0.0
    if direction == "BUY":
        return (price - entry) / initial_risk
    return (entry - price) / initial_risk


def _ratchet_sl(direction: TradeDirection, current_sl: float, new_sl: float) -> float:
    if direction == "BUY":
        return max(current_sl, new_sl)
    return min(current_sl, new_sl)


def _breakeven_sl(direction: TradeDirection, entry: float, atr: float, buffer_atr: float) -> float:
    buffer = buffer_atr * atr
    if direction == "BUY":
        return entry + buffer
    return entry - buffer


@dataclass
class CspaExitTracker:
    """1 ポジション分の CSPA 建値 / トレール状態（L5 と同一ルール）。"""

    direction: TradeDirectionStr
    entry: float
    initial_sl: float
    take_profit: float
    atr: float
    be_enabled: bool = CSPA_BE_ENABLED
    trail_enabled: bool = CSPA_TRAIL_ENABLED
    be_arm_mfe_r: float = CSPA_BE_ARM_MFE_R
    be_trigger_mfe_r: float = CSPA_BE_TRIGGER_MFE_R
    be_pullback_close_r: float = CSPA_BE_PULLBACK_CLOSE_R
    be_rhythm_max_bars: int = CSPA_BE_RHYTHM_MAX_BARS
    trail_atr_mult: float = CSPA_TRAIL_ATR_MULT
    be_buffer_atr: float = CSPA_BE_BUFFER_ATR
    current_sl: float = 0.0
    peak_favorable: float = 0.0
    extension_armed: bool = False
    sl_at_breakeven: bool = False
    bars_since_entry: int = 0

    def __post_init__(self) -> None:
        self.direction = self.direction.upper()  # type: ignore[assignment]
        if self.current_sl == 0.0:
            self.current_sl = self.initial_sl
        if self.peak_favorable == 0.0:
            self.peak_favorable = self.entry

    @property
    def initial_risk(self) -> float:
        return abs(self.entry - self.initial_sl)

    @classmethod
    def from_signal(
        cls,
        *,
        direction: str,
        entry: float,
        initial_sl: float,
        take_profit: float,
        exit_fields: dict[str, Any],
    ) -> CspaExitTracker:
        return cls(
            direction=direction.upper(),  # type: ignore[arg-type]
            entry=entry,
            initial_sl=initial_sl,
            take_profit=take_profit,
            atr=float(exit_fields.get("exit_atr", 0.0)),
            be_enabled=bool(int(exit_fields.get("exit_be_enabled", int(CSPA_BE_ENABLED)))),
            trail_enabled=bool(int(exit_fields.get("exit_trail_enabled", int(CSPA_TRAIL_ENABLED)))),
            be_arm_mfe_r=float(exit_fields.get("exit_be_arm_mfe_r", CSPA_BE_ARM_MFE_R)),
            be_trigger_mfe_r=float(exit_fields.get("exit_be_trigger_mfe_r", CSPA_BE_TRIGGER_MFE_R)),
            be_pullback_close_r=float(exit_fields.get("exit_be_pullback_close_r", CSPA_BE_PULLBACK_CLOSE_R)),
            be_rhythm_max_bars=int(exit_fields.get("exit_be_rhythm_max_bars", CSPA_BE_RHYTHM_MAX_BARS)),
            trail_atr_mult=float(exit_fields.get("exit_trail_atr_mult", CSPA_TRAIL_ATR_MULT)),
            be_buffer_atr=float(exit_fields.get("exit_be_buffer_atr", CSPA_BE_BUFFER_ATR)),
        )

    def on_bar(self, high: float, low: float, close: float) -> float:
        """確定バー 1 本分を処理し、更新後の SL を返す（`track_cspa_trade_outcome` と同一）。"""
        self.bars_since_entry += 1
        risk = self.initial_risk
        if risk <= 0:
            return self.current_sl

        trail_atr = max(self.atr, risk * 0.25)
        be_sl = _breakeven_sl(self.direction, self.entry, trail_atr, self.be_buffer_atr)

        if self.direction == "BUY":
            bar_mfe_r = (high - self.entry) / risk
            self.peak_favorable = max(self.peak_favorable, high)
        else:
            bar_mfe_r = (self.entry - low) / risk
            self.peak_favorable = min(self.peak_favorable, low)

        if self.be_enabled:
            if bar_mfe_r >= self.be_arm_mfe_r:
                self.extension_armed = True
            close_r = _profit_r(self.direction, self.entry, close, risk)
            rhythm_window = self.bars_since_entry <= self.be_rhythm_max_bars
            if bar_mfe_r >= self.be_trigger_mfe_r:
                self.current_sl = _ratchet_sl(self.direction, self.current_sl, be_sl)
                self.sl_at_breakeven = True
            elif self.extension_armed and rhythm_window and close_r <= self.be_pullback_close_r:
                self.current_sl = _ratchet_sl(self.direction, self.current_sl, be_sl)
                self.sl_at_breakeven = True

        if self.trail_enabled and self.sl_at_breakeven:
            if self.direction == "BUY":
                trail_sl = self.peak_favorable - self.trail_atr_mult * trail_atr
                trail_sl = max(trail_sl, be_sl)
            else:
                trail_sl = self.peak_favorable + self.trail_atr_mult * trail_atr
                trail_sl = min(trail_sl, be_sl)
            self.current_sl = _ratchet_sl(self.direction, self.current_sl, trail_sl)

        return self.current_sl
