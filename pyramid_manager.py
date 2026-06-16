"""
pyramid_manager.py — ピラミッディング + トレーリング SL 統合モジュール

L5 未来追跡（バー High/Low）と統合し、積み増し・ratchet SL・全決済を管理する。
BACKTEST_MODE でも同一ロジック（確定バーのみ参照、ルックアヘッドなし）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from audit.broker_costs import apply_commission_sl_floor, min_net_profit_sl

PIP_SIZE = 0.0001
LSFC_SETUP_TYPE = "LONDON_SWEEP_FAILURE_CONTINUATION"
ALS_SETUP_TYPE = "ASIAN_SESSION_LIQUIDITY_SWEEP"
FVG_SETUP_TYPE = "FVG_FILL"
TREF_SETUP_TYPE = "TOKYO_RANGE_EXPANSION_FAILURE"
WYCKOFF_REVERSAL_SETUP_TYPE = "WYCKOFF_REVERSAL"
WYCKOFF_SPRING_SETUP_TYPE = "WYCKOFF_SPRING"
CSPA_SETUP_TYPE = "CSPA"
TTM_SETUP_TYPE = "TTM_LIQUIDITY_EVENT"
SMRS_SETUP_TYPE = "SMRS"

# --- ストラテジー別 L5 ピラミッド既定（コードレベル） ---
# 環境変数 PYRAMID_<STRATEGY>=0/1 で個別上書き。PYRAMID_ENABLED=0 で全体 OFF。
PYRAMID_STRATEGY_DEFAULTS: dict[str, bool] = {
    LSFC_SETUP_TYPE: True,
    ALS_SETUP_TYPE: False,
    FVG_SETUP_TYPE: False,
    TREF_SETUP_TYPE: True,
    WYCKOFF_REVERSAL_SETUP_TYPE: False,
    WYCKOFF_SPRING_SETUP_TYPE: False,
    CSPA_SETUP_TYPE: True,
    TTM_SETUP_TYPE: False,
    SMRS_SETUP_TYPE: False,
}

STRATEGY_PYRAMID_ENV_VARS: dict[str, str] = {
    LSFC_SETUP_TYPE: "PYRAMID_LSFC",
    ALS_SETUP_TYPE: "PYRAMID_ALS",
    FVG_SETUP_TYPE: "PYRAMID_FVG",
    TREF_SETUP_TYPE: "PYRAMID_TREF",
    WYCKOFF_REVERSAL_SETUP_TYPE: "PYRAMID_WYCKOFF",
    WYCKOFF_SPRING_SETUP_TYPE: "PYRAMID_WYCKOFF",
    CSPA_SETUP_TYPE: "PYRAMID_CSPA",
    TTM_SETUP_TYPE: "PYRAMID_TTM",
    SMRS_SETUP_TYPE: "PYRAMID_SMRS",
}


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return None


def pyramid_env_var_for_setup_type(setup_type: str) -> str | None:
    """ストラテジーに対応する環境変数名（例: PYRAMID_LSFC）。"""
    return STRATEGY_PYRAMID_ENV_VARS.get(setup_type.strip())


def default_pyramid_enabled_for_strategy(setup_type: str) -> bool:
    """コード既定のピラミッド ON/OFF（環境変数未設定時）。"""
    return PYRAMID_STRATEGY_DEFAULTS.get(setup_type.strip(), False)


def resolve_pyramid_enabled(setup_type: str) -> bool:
    """1 ストラテジー分の有効判定（PYRAMID_ENABLED 全体スイッチは含まない）。"""
    key = setup_type.strip()
    if key not in PYRAMID_STRATEGY_DEFAULTS:
        return False
    env_var = STRATEGY_PYRAMID_ENV_VARS.get(key)
    if env_var:
        override = _env_flag(env_var)
        if override is not None:
            return override
    return PYRAMID_STRATEGY_DEFAULTS[key]


def get_pyramid_strategy_status() -> list[dict[str, Any]]:
    """全ストラテジーの既定 / 環境変数 / 実効状態を返す（監査・ヘルス用）。"""
    global_off = _env_flag("PYRAMID_ENABLED") is False
    rows: list[dict[str, Any]] = []
    for setup_type in PYRAMID_STRATEGY_DEFAULTS:
        env_var = STRATEGY_PYRAMID_ENV_VARS.get(setup_type, "")
        override = _env_flag(env_var) if env_var else None
        rows.append(
            {
                "setup_type": setup_type,
                "default_enabled": PYRAMID_STRATEGY_DEFAULTS[setup_type],
                "env_var": env_var,
                "env_override": override,
                "effective_enabled": False if global_off else resolve_pyramid_enabled(setup_type),
            }
        )
    return rows


def is_pyramid_enabled_for_pending(pending: Any) -> bool:
    """
    PendingEvaluation 単位のピラミッド判定。

    TTMS: TTM_EV_PYRAMID_TOP20=1 かつ ev_rank >= 0.80 (Top 20%) のみ ON。
    その他: is_pyramid_enabled(setup_type)。
    """
    if _env_flag("PYRAMID_ENABLED") is False:
        return False
    setup_type = str(getattr(pending, "setup_type", "") or "").strip()
    if setup_type == TTM_SETUP_TYPE:
        from strategies.ttm_bayes_ev import is_ttm_top20_ev_rank, is_ttm_top20_pyramid_enabled

        if not is_ttm_top20_pyramid_enabled():
            return False
        return is_ttm_top20_ev_rank(float(getattr(pending, "ttm_ev_rank", 0.0) or 0.0))
    return is_pyramid_enabled(setup_type)


def is_pyramid_enabled(setup_type: str | None = None) -> bool:
    """
    ピラミッド追跡の有効判定。

    優先順位:
      1. PYRAMID_ENABLED=0/off → 全ストラテジー OFF
      2. PYRAMID_<STRATEGY>=0/1 → 当該ストラテジーのみ上書き
      3. PYRAMID_STRATEGY_DEFAULTS → コード既定

    既定 ON: LSFC, TREF / 既定 OFF: ALS, FVG, WYCKOFF_SPRING
    （FVG / WYCKOFF は PYRAMID_FVG=1 / PYRAMID_WYCKOFF=1 で opt-in）
    """
    if _env_flag("PYRAMID_ENABLED") is False:
        return False
    if setup_type is None:
        return True
    return resolve_pyramid_enabled(setup_type)


DEFAULT_MAX_PYRAMID_LAYERS = 3
DEFAULT_TREF_MAX_PYRAMID_LAYERS = 3


def resolve_max_pyramid_layers(setup_type: str | None = None) -> int:
    """
    ピラミッド総ポジション数上限（初回エントリー含む）。

    - 既定: **3**（初回 + 追加2段 → CSV `pyramid_layers` 最大 **2**）
    - TREF: 同上（v4.1 — `PYRAMID_TREF_MAX_LAYERS` で上書き可）
    - 全体: `PYRAMID_MAX_LAYERS` で LSFC 等の既定を上書き
    """
    default_raw = os.environ.get("PYRAMID_MAX_LAYERS", "").strip()
    default = int(default_raw) if default_raw else DEFAULT_MAX_PYRAMID_LAYERS
    if setup_type is not None and setup_type.strip() == TREF_SETUP_TYPE:
        tref_raw = os.environ.get("PYRAMID_TREF_MAX_LAYERS", "").strip()
        if tref_raw:
            return max(2, int(tref_raw))
        return DEFAULT_TREF_MAX_PYRAMID_LAYERS
    return max(2, default)


def is_backtest_mode() -> bool:
    return os.environ.get("BACKTEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class PyramidPosition:
    """個別ポジション（エントリー1回分）の管理。"""

    entry_price: float
    lot_size: float
    original_sl: float
    current_sl: float
    entry_order: int
    profit_r: float = 0.0


@dataclass
class PyramidTradeResult:
    """ピラミッド込み L5 追跡結果。"""

    result: str
    profit_r: float
    profit_pips: float
    holding_minutes: int
    pyramid_layers: int
    pyramid_entry_prices: list[float]
    pyramid_lot_sizes: list[float]
    final_sl_at_close: float
    peak_unrealized_r: float


@dataclass
class PyramidManager:
    """ピラミッディング全体の管理。"""

    trade_id: str
    direction: str
    atr: float
    base_risk: float
    breakeven_price: float
    take_profit: float
    initial_lot: float
    initial_stop_loss: float
    max_pyramid_layers: int = 3
    daily_dd_remaining_percent: float = 5.0
    symbol: str = ""
    tick_size: float = 0.0
    tick_value: float = 0.0
    positions: list[PyramidPosition] = field(default_factory=list)
    peak_unrealized_r: float = 0.0
    peak_favorable_price: float = 0.0
    _sl_at_breakeven: bool = False

    def __post_init__(self) -> None:
        self.direction = self.direction.upper()
        if not self.positions:
            self.positions = [
                PyramidPosition(
                    entry_price=self.breakeven_price,
                    lot_size=self.initial_lot,
                    original_sl=self.initial_stop_loss,
                    current_sl=self.initial_stop_loss,
                    entry_order=1,
                )
            ]
        self.peak_favorable_price = self.breakeven_price

    @property
    def layer_count(self) -> int:
        return len(self.positions)

    @property
    def pyramid_layers(self) -> int:
        """追加段数（初回除く）。"""
        return max(0, self.layer_count - 1)

    def total_lot_size(self) -> float:
        return sum(p.lot_size for p in self.positions)

    def min_net_profit_sl(self) -> float:
        """Fintokei 手数料 + 微益バッファ込みの SL 下限/上限。"""
        legs = [(p.entry_price, p.lot_size) for p in self.positions]
        return min_net_profit_sl(
            self.direction,
            legs,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            symbol=self.symbol,
        )

    def _commission_sl_floor(self) -> float:
        return self.min_net_profit_sl()

    def average_entry_price(self) -> float:
        total_lot = sum(p.lot_size for p in self.positions)
        if total_lot <= 0:
            return self.breakeven_price
        return sum(p.entry_price * p.lot_size for p in self.positions) / total_lot

    def portfolio_unrealized_r(self, price: float) -> float:
        if self.base_risk <= 0:
            return 0.0
        avg = self.average_entry_price()
        if self.direction == "BUY":
            return (price - avg) / self.base_risk
        return (avg - price) / self.base_risk

    def _all_sl_at_breakeven(self) -> bool:
        floor = self._commission_sl_floor()
        if self.direction == "BUY":
            return all(p.current_sl >= floor for p in self.positions)
        return all(p.current_sl <= floor for p in self.positions)

    def move_all_sl_to_breakeven(self) -> None:
        """条件B: 全 SL を手数料込み建値以上/以下へ。"""
        floor = self._commission_sl_floor()
        for pos in self.positions:
            if self.direction == "BUY":
                pos.current_sl = max(pos.current_sl, floor)
            else:
                pos.current_sl = min(pos.current_sl, floor)
        self._sl_at_breakeven = self._all_sl_at_breakeven()

    def _ratchet_sl(self, new_sl: float) -> None:
        """SL は利益方向にのみ更新（ratchet）+ 手数料込み微益下限。"""
        floor = self._commission_sl_floor()
        new_sl = apply_commission_sl_floor(self.direction, new_sl, floor)
        for pos in self.positions:
            if self.direction == "BUY":
                pos.current_sl = max(pos.current_sl, new_sl)
            else:
                pos.current_sl = min(pos.current_sl, new_sl)

    def update_trailing_from_peak(self) -> None:
        """最高含み益更新時: peak ∓ 1.0×ATR、ただし手数料込み微益 SL 下限を下回らない。"""
        floor = self._commission_sl_floor()
        if self.direction == "BUY":
            trail = self.peak_favorable_price - 1.0 * self.atr
            trail = max(trail, floor)
            self._ratchet_sl(trail)
        else:
            trail = self.peak_favorable_price + 1.0 * self.atr
            trail = min(trail, floor)
            self._ratchet_sl(trail)

    def update_peak(self, high: float, low: float, close: float) -> None:
        if self.direction == "BUY":
            favorable = high
        else:
            favorable = low
        unr = self.portfolio_unrealized_r(favorable)
        if unr > self.peak_unrealized_r:
            self.peak_unrealized_r = unr
            self.peak_favorable_price = favorable
            self.update_trailing_from_peak()
        _ = close

    def pyramid_lot_for_next_layer(self) -> float:
        """条件C: 2段目=50%, 3段目=25%（初回比）。"""
        layer = self.layer_count + 1
        return self.initial_lot * (0.5 ** (layer - 1))

    def can_add_pyramid(self, price: float, daily_dd_remaining: float | None = None) -> tuple[bool, str]:
        dd_rem = daily_dd_remaining if daily_dd_remaining is not None else self.daily_dd_remaining_percent

        if self.layer_count >= self.max_pyramid_layers:
            return False, "MAX_LAYERS"

        if not self._all_sl_at_breakeven():
            return False, "SL_NOT_AT_BREAKEVEN"

        if self.portfolio_unrealized_r(price) < 1.0:
            return False, "UNREALIZED_R_BELOW_1"

        if dd_rem < 2.0:
            return False, "DD_REMAINING_LOW"

        return True, "OK"

    def add_pyramid_layer(self, price: float) -> None:
        """積み増し実行 + 全 SL を直前価格 ∓ 0.5×ATR へ ratchet。"""
        if self.direction == "BUY":
            self._ratchet_sl(price - 0.5 * self.atr)
        else:
            self._ratchet_sl(price + 0.5 * self.atr)

        lot = self.pyramid_lot_for_next_layer()
        unified = self.unified_stop_loss()
        self.positions.append(
            PyramidPosition(
                entry_price=price,
                lot_size=lot,
                original_sl=unified,
                current_sl=unified,
                entry_order=self.layer_count + 1,
            )
        )

    def unified_stop_loss(self) -> float:
        """全ポジション共通 SL（最も保守的な値）。"""
        if self.direction == "BUY":
            return max(p.current_sl for p in self.positions)
        return min(p.current_sl for p in self.positions)

    def sl_hit_on_bar(self, high: float, low: float) -> bool:
        sl = self.unified_stop_loss()
        if self.direction == "BUY":
            return low <= sl
        return high >= sl

    def tp_hit_on_bar(self, high: float, low: float) -> bool:
        if self.direction == "BUY":
            return high >= self.take_profit
        return low <= self.take_profit

    def close_at_price(self, exit_price: float) -> tuple[str, float, float]:
        """全ポジション同時決済 → 加重 R / pips。"""
        if self.base_risk <= 0 or self.initial_lot <= 0:
            return "LOSS", -1.0, 0.0

        total_pnl_price = 0.0
        for pos in self.positions:
            if self.direction == "BUY":
                total_pnl_price += (exit_price - pos.entry_price) * pos.lot_size
            else:
                total_pnl_price += (pos.entry_price - exit_price) * pos.lot_size

        denom = self.base_risk * self.initial_lot
        profit_r = total_pnl_price / denom if denom > 0 else 0.0
        profit_r = max(-1.0, min(2.4, profit_r))
        profit_pips = total_pnl_price / (self.initial_lot * PIP_SIZE) if self.initial_lot > 0 else 0.0
        result = "WIN" if profit_r > 0 else "LOSS"
        return result, profit_r, profit_pips

    def to_result_fields(self, exit_price: float, holding: int, result: str, profit_r: float, profit_pips: float) -> PyramidTradeResult:
        return PyramidTradeResult(
            result=result,
            profit_r=profit_r,
            profit_pips=profit_pips,
            holding_minutes=holding,
            pyramid_layers=self.pyramid_layers,
            pyramid_entry_prices=[round(p.entry_price, 5) for p in self.positions],
            pyramid_lot_sizes=[round(p.lot_size, 4) for p in self.positions],
            final_sl_at_close=round(self.unified_stop_loss(), 5),
            peak_unrealized_r=round(self.peak_unrealized_r, 4),
        )


def empty_pyramid_result(
    stop_loss: float,
    result: str = "LOSS",
    profit_r: float = 0.0,
    profit_pips: float = 0.0,
    holding: int = 0,
) -> PyramidTradeResult:
    return PyramidTradeResult(
        result=result,
        profit_r=profit_r,
        profit_pips=profit_pips,
        holding_minutes=holding,
        pyramid_layers=0,
        pyramid_entry_prices=[],
        pyramid_lot_sizes=[],
        final_sl_at_close=round(stop_loss, 5),
        peak_unrealized_r=0.0,
    )


def pyramid_result_to_record_fields(result: PyramidTradeResult) -> dict[str, Any]:
    return {
        "pyramid_layers": result.pyramid_layers,
        "pyramid_entry_prices": json.dumps(result.pyramid_entry_prices, ensure_ascii=False),
        "pyramid_lot_sizes": json.dumps(result.pyramid_lot_sizes, ensure_ascii=False),
        "final_sl_at_close": result.final_sl_at_close,
        "peak_unrealized_r": result.peak_unrealized_r,
    }


def simulate_pyramid_on_bars(
    df: pd.DataFrame,
    start_index: int,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    bar_minutes: int,
    initial_lot: float,
    atr: float,
    daily_dd_remaining_percent: float,
    *,
    max_pyramid_layers: int = 3,
    trade_id: str = "",
    force_close_at_timeout: bool = False,
    timeout_server_hour: int = 0,
    entry_timestamp: pd.Timestamp | None = None,
    max_holding_bars: int = 48,
) -> PyramidTradeResult:
    """
    バー逐次シミュレーション（ルックアヘッドなし）。

    各バー i では df.iloc[i] の OHLC のみ使用。積み増し判定はバー終値。
    """
    from main_platform import (
        _compute_session_timeout_deadline,
        _force_close_pnl,
    )

    from strategies.bt_ohlcv import as_ohlcv, find_first_bar_at_or_after_np, normalize_ts_ns

    ohlcv = as_ohlcv(df)
    base_risk = abs(entry - stop_loss)
    if base_risk <= 0 or initial_lot <= 0:
        return empty_pyramid_result(stop_loss)

    mgr = PyramidManager(
        trade_id=trade_id,
        direction=direction,
        atr=max(atr, base_risk * 0.01),
        base_risk=base_risk,
        breakeven_price=entry,
        take_profit=take_profit,
        initial_lot=initial_lot,
        initial_stop_loss=stop_loss,
        max_pyramid_layers=max_pyramid_layers,
        daily_dd_remaining_percent=daily_dd_remaining_percent,
    )

    session_timeout_idx: int | None = None
    if force_close_at_timeout and timeout_server_hour > 0 and entry_timestamp is not None:
        deadline = _compute_session_timeout_deadline(entry_timestamp, timeout_server_hour)
        session_timeout_idx = find_first_bar_at_or_after_np(
            ohlcv,
            start_index,
            normalize_ts_ns(deadline),
        )

    max_holding_end = min(start_index + max_holding_bars, ohlcv.length - 1)
    end_index = (
        min(max_holding_end, session_timeout_idx)
        if session_timeout_idx is not None
        else max_holding_end
    )

    for i in range(start_index + 1, end_index + 1):
        high = float(ohlcv.high[i])
        low = float(ohlcv.low[i])
        close = float(ohlcv.close[i])
        elapsed = (i - start_index) * bar_minutes

        sl_hit = mgr.sl_hit_on_bar(high, low)
        tp_hit = mgr.tp_hit_on_bar(high, low)

        if sl_hit and tp_hit:
            exit_price = mgr.unified_stop_loss()
            res, pr, pp = mgr.close_at_price(exit_price)
            return mgr.to_result_fields(exit_price, elapsed, res, min(pr, -0.01), pp)

        if sl_hit:
            exit_price = mgr.unified_stop_loss()
            res, pr, pp = mgr.close_at_price(exit_price)
            return mgr.to_result_fields(exit_price, elapsed, res, pr, pp)

        if tp_hit:
            res, pr, pp = mgr.close_at_price(take_profit)
            return mgr.to_result_fields(take_profit, elapsed, "WIN", max(pr, 2.0), pp)

        mgr.update_peak(high, low, close)

        if mgr.portfolio_unrealized_r(close) >= 1.0 and not mgr._all_sl_at_breakeven():
            mgr.move_all_sl_to_breakeven()

        can_add, _ = mgr.can_add_pyramid(close, daily_dd_remaining_percent)
        if can_add:
            mgr.add_pyramid_layer(close)

        if session_timeout_idx is not None and i == session_timeout_idx:
            res, pr, pp = mgr.close_at_price(close)
            return mgr.to_result_fields(close, elapsed, res, pr, pp)

    last_close = float(ohlcv.close[end_index])
    holding = (end_index - start_index) * bar_minutes
    res, pr, pp = mgr.close_at_price(last_close)
    return mgr.to_result_fields(last_close, holding, res, pr, pp)


def track_with_pyramid(
    df: pd.DataFrame,
    start_index: int,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    bar_minutes: int,
    initial_lot: float,
    atr: float,
    daily_dd_remaining_percent: float,
    *,
    setup_type: str | None = None,
    skip_strategy_enable_check: bool = False,
    **kwargs: Any,
) -> PyramidTradeResult:
    """feature_engineering 統合用エントリポイント。"""
    if not skip_strategy_enable_check and not is_pyramid_enabled(setup_type):
        from main_platform import track_trade_outcome

        outcome = track_trade_outcome(
            df,
            start_index,
            direction,
            entry,
            stop_loss,
            take_profit,
            bar_minutes,
            force_close_at_timeout=kwargs.get("force_close_at_timeout", False),
            timeout_server_hour=kwargs.get("timeout_server_hour", 0),
            entry_timestamp=kwargs.get("entry_timestamp"),
        )
        return empty_pyramid_result(
            stop_loss,
            result=outcome.result,
            profit_r=outcome.profit_r,
            profit_pips=outcome.profit_pips,
            holding=outcome.holding_minutes,
        )

    return simulate_pyramid_on_bars(
        df,
        start_index,
        direction,
        entry,
        stop_loss,
        take_profit,
        bar_minutes,
        initial_lot,
        atr,
        daily_dd_remaining_percent,
        trade_id=str(kwargs.get("trade_id", "")),
        force_close_at_timeout=kwargs.get("force_close_at_timeout", False),
        timeout_server_hour=kwargs.get("timeout_server_hour", 0),
        entry_timestamp=kwargs.get("entry_timestamp"),
        max_holding_bars=kwargs.get("max_holding_bars", 48),
        max_pyramid_layers=resolve_max_pyramid_layers(setup_type),
    )
