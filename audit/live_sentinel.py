"""
audit/live_sentinel.py — Live Sentinel (リアルタイム生命維持装置)

Fintokei 等のライブ口座向けセーフティネット:
  1. ブローカーサーバー日次リセット (TimeCurrent / server_time 基準)
  2. Intraday Equity Terminator — 含み損 (Floating DD) 強制決済 + エントリーロック
  3. ロールオーバー時間帯フィルター + スプレッドプロテクション

判定式 (Floating DD):
    (day_high_balance - equity) / day_start_balance × 100
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from audit.risk_manager import MAX_DAILY_DD_PCT, STARTING_EQUITY

logger = logging.getLogger("live_sentinel")

REASON_SENTINEL_TERMINATOR = "LIVE_SENTINEL_TERMINATOR"
REASON_SENTINEL_ENTRY_LOCK = "LIVE_SENTINEL_ENTRY_LOCK"
REASON_SENTINEL_ROLLOVER = "LIVE_SENTINEL_ROLLOVER"
REASON_SENTINEL_SPREAD = "LIVE_SENTINEL_SPREAD"

# Fintokei 失格ライン 4.5% 手前の安全バッファ (3.5–4.0% 推奨)
FLOATING_DD_TRIGGER_PCT = float(os.getenv("LIVE_SENTINEL_FLOATING_DD_TRIGGER", "3.75"))
FLOATING_DD_WARN_PCT = float(os.getenv("LIVE_SENTINEL_FLOATING_DD_WARN", "3.5"))
DAILY_DD_DISQUALIFY_PCT = float(os.getenv("LIVE_SENTINEL_DAILY_DISQUALIFY", "4.5"))

ROLLOVER_START_HOUR = int(os.getenv("LIVE_SENTINEL_ROLLOVER_START_HOUR", "23"))
ROLLOVER_START_MINUTE = int(os.getenv("LIVE_SENTINEL_ROLLOVER_START_MIN", "55"))
ROLLOVER_END_HOUR = int(os.getenv("LIVE_SENTINEL_ROLLOVER_END_HOUR", "0"))
ROLLOVER_END_MINUTE = int(os.getenv("LIVE_SENTINEL_ROLLOVER_END_MIN", "10"))

MAX_SPREAD_POINTS = int(os.getenv("LIVE_SENTINEL_MAX_SPREAD_POINTS", "30"))


def is_live_sentinel_enabled() -> bool:
    raw = os.getenv("LIVE_SENTINEL_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "disabled")


def parse_server_time(value: str | datetime) -> datetime:
    """MT5 server_time / bar_time 文字列を datetime へ。"""
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(text.replace("T", " "))


def server_trading_day(server_time: datetime) -> date:
    """ブローカーサーバー日付 (MT5 TimeCurrent 相当)。"""
    return server_time.date()


def _minutes_since_midnight(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def is_in_rollover_window(
    server_time: datetime,
    *,
    start_hour: int = ROLLOVER_START_HOUR,
    start_minute: int = ROLLOVER_START_MINUTE,
    end_hour: int = ROLLOVER_END_HOUR,
    end_minute: int = ROLLOVER_END_MINUTE,
) -> bool:
    """
    サーバー時間のロールオーバー帯 (既定 23:55–00:10) を判定。
    日付を跨ぐ区間を正しく扱う。
    """
    now_min = _minutes_since_midnight(server_time)
    start_min = start_hour * 60 + start_minute
    end_min = end_hour * 60 + end_minute
    if start_min <= end_min:
        return start_min <= now_min <= end_min
    return now_min >= start_min or now_min <= end_min


def floating_dd_percent(
    day_start_balance: float,
    day_high_balance: float,
    equity: float,
) -> float:
    """
    当日リアルタイム含み損 (Floating DD) [%]。

    (当日最高 Balance - 現在 Equity) / 日初 Balance × 100
    """
    if day_start_balance <= 0.0:
        return 0.0
    floating_loss = max(0.0, day_high_balance - equity)
    return floating_loss / day_start_balance * 100.0


def daily_dd_remaining_percent(
    day_start_equity: float,
    equity: float,
    *,
    limit_pct: float = MAX_DAILY_DD_PCT,
) -> float:
    """日次 DD 残量 [%] — サーバー日初 Equity 基準。"""
    if day_start_equity <= 0.0:
        return limit_pct
    used = max(0.0, (day_start_equity - equity) / day_start_equity * 100.0)
    return max(0.0, limit_pct - used)


@dataclass
class LiveSentinelState:
    """セッション永続状態 (LivePipelineState / EA 双方で同期可能)。"""

    server_trading_day: date | None = None
    day_start_balance: float = STARTING_EQUITY
    day_start_equity: float = STARTING_EQUITY
    day_high_balance: float = STARTING_EQUITY
    day_high_equity: float = STARTING_EQUITY
    daily_dd_remaining_pct: float = MAX_DAILY_DD_PCT
    floating_dd_pct: float = 0.0
    entry_locked: bool = False
    terminator_fired: bool = False
    spread_hold_active: bool = False
    last_server_time: datetime | None = None
    last_spread_points: int | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(cls) -> LiveSentinelState:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_trading_day": self.server_trading_day.isoformat() if self.server_trading_day else None,
            "day_start_balance": self.day_start_balance,
            "day_start_equity": self.day_start_equity,
            "day_high_balance": self.day_high_balance,
            "day_high_equity": self.day_high_equity,
            "daily_dd_remaining_pct": round(self.daily_dd_remaining_pct, 4),
            "floating_dd_pct": round(self.floating_dd_pct, 4),
            "entry_locked": self.entry_locked,
            "terminator_fired": self.terminator_fired,
            "spread_hold_active": self.spread_hold_active,
            "last_server_time": self.last_server_time.isoformat(sep=" ") if self.last_server_time else None,
            "last_spread_points": self.last_spread_points,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class SentinelVerdict:
    """1 tick あたりの Sentinel 判定結果。"""

    entry_allowed: bool
    panic_close: bool
    rollover_block: bool
    spread_block: bool
    entry_locked: bool
    floating_dd_pct: float
    daily_dd_remaining_pct: float
    message: str
    tags: tuple[str, ...] = ()
    log_level: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_allowed": self.entry_allowed,
            "panic_close": self.panic_close,
            "rollover_block": self.rollover_block,
            "spread_block": self.spread_block,
            "entry_locked": self.entry_locked,
            "floating_dd_pct": round(self.floating_dd_pct, 4),
            "daily_dd_remaining_pct": round(self.daily_dd_remaining_pct, 4),
            "message": self.message,
            "tags": list(self.tags),
            "log_level": self.log_level,
        }


def _reset_server_day(
    state: LiveSentinelState,
    server_time: datetime,
    balance: float,
    equity: float,
) -> None:
    """サーバー 00:00 相当 — 日次 DD 残量を MAX_DAILY_DD_PCT へ完全リセット。"""
    trading_day = server_trading_day(server_time)
    state.server_trading_day = trading_day
    state.day_start_balance = balance
    state.day_start_equity = equity
    state.day_high_balance = balance
    state.day_high_equity = equity
    state.daily_dd_remaining_pct = MAX_DAILY_DD_PCT
    state.floating_dd_pct = 0.0
    state.entry_locked = False
    state.terminator_fired = False
    state.spread_hold_active = False
    state.tags = []
    state.last_server_time = server_time
    logger.info(
        "LIVE_SENTINEL daily reset | server_day=%s balance=%.2f equity=%.2f remaining=%.2f%%",
        trading_day.isoformat(),
        balance,
        equity,
        MAX_DAILY_DD_PCT,
    )


def _update_intraday_extremes(
    state: LiveSentinelState,
    balance: float,
    equity: float,
) -> None:
    state.day_high_balance = max(state.day_high_balance, balance)
    state.day_high_equity = max(state.day_high_equity, equity)
    state.floating_dd_pct = floating_dd_percent(
        state.day_start_balance,
        state.day_high_balance,
        equity,
    )
    state.daily_dd_remaining_pct = daily_dd_remaining_percent(
        state.day_start_equity,
        equity,
    )


def evaluate_live_sentinel(
    state: LiveSentinelState,
    server_time: datetime,
    balance: float,
    equity: float,
    *,
    spread_points: int | None = None,
    max_spread_points: int = MAX_SPREAD_POINTS,
    floating_trigger_pct: float = FLOATING_DD_TRIGGER_PCT,
    enabled: bool | None = None,
) -> SentinelVerdict:
    """
    Live Sentinel メインエントリ — OnTick / trade_signal 前に呼ぶ。

    Returns:
        SentinelVerdict — panic_close=True なら即全決済、entry_allowed=False なら新規拒否。
    """
    if enabled is None:
        enabled = is_live_sentinel_enabled()
    if not enabled:
        return SentinelVerdict(
            entry_allowed=True,
            panic_close=False,
            rollover_block=False,
            spread_block=False,
            entry_locked=False,
            floating_dd_pct=0.0,
            daily_dd_remaining_pct=state.daily_dd_remaining_pct,
            message="Live Sentinel disabled",
        )

    trading_day = server_trading_day(server_time)
    if state.server_trading_day != trading_day:
        _reset_server_day(state, server_time, balance, equity)
    else:
        state.last_server_time = server_time
        _update_intraday_extremes(state, balance, equity)

    tags: list[str] = []
    rollover_block = is_in_rollover_window(server_time)
    spread_block = False
    spread_hold = False

    if spread_points is not None:
        state.last_spread_points = spread_points
        if spread_points > max_spread_points:
            spread_block = True
            spread_hold = True
            tags.append(REASON_SENTINEL_SPREAD)

    state.spread_hold_active = spread_hold

    if state.entry_locked or state.terminator_fired:
        msg = (
            f"LIVE_SENTINEL entry lock active until server 00:00 | "
            f"floating={state.floating_dd_pct:.2f}% remaining={state.daily_dd_remaining_pct:.2f}%"
        )
        logger.error(msg)
        return SentinelVerdict(
            entry_allowed=False,
            panic_close=False,
            rollover_block=rollover_block,
            spread_block=spread_block,
            entry_locked=True,
            floating_dd_pct=state.floating_dd_pct,
            daily_dd_remaining_pct=state.daily_dd_remaining_pct,
            message=msg,
            tags=(REASON_SENTINEL_ENTRY_LOCK, *tags),
            log_level="error",
        )

    if state.floating_dd_pct >= floating_trigger_pct:
        state.entry_locked = True
        state.terminator_fired = True
        tags.append(REASON_SENTINEL_TERMINATOR)
        msg = (
            f"LIVE_SENTINEL TERMINATOR | floating_dd={state.floating_dd_pct:.2f}% "
            f">= trigger={floating_trigger_pct:.2f}% | "
            f"high_balance={state.day_high_balance:.2f} equity={equity:.2f} | "
            f"PANIC CLOSE + entry lock until server midnight"
        )
        logger.error(msg)
        return SentinelVerdict(
            entry_allowed=False,
            panic_close=True,
            rollover_block=rollover_block,
            spread_block=spread_block,
            entry_locked=True,
            floating_dd_pct=state.floating_dd_pct,
            daily_dd_remaining_pct=state.daily_dd_remaining_pct,
            message=msg,
            tags=tuple(tags),
            log_level="error",
        )

    if rollover_block:
        tags.append(REASON_SENTINEL_ROLLOVER)
        msg = (
            f"LIVE_SENTINEL rollover window block | server_time={server_time.strftime('%H:%M')} | "
            f"floating={state.floating_dd_pct:.2f}%"
        )
        logger.warning(msg)
        return SentinelVerdict(
            entry_allowed=False,
            panic_close=False,
            rollover_block=True,
            spread_block=spread_block,
            entry_locked=False,
            floating_dd_pct=state.floating_dd_pct,
            daily_dd_remaining_pct=state.daily_dd_remaining_pct,
            message=msg,
            tags=tuple(tags),
            log_level="warning",
        )

    if spread_block:
        msg = (
            f"LIVE_SENTINEL spread block | spread={spread_points} > max={max_spread_points} | "
            f"new entries halted; logic-side close held"
        )
        logger.warning(msg)
        return SentinelVerdict(
            entry_allowed=False,
            panic_close=False,
            rollover_block=False,
            spread_block=True,
            entry_locked=False,
            floating_dd_pct=state.floating_dd_pct,
            daily_dd_remaining_pct=state.daily_dd_remaining_pct,
            message=msg,
            tags=tuple(tags),
            log_level="warning",
        )

    if state.floating_dd_pct >= FLOATING_DD_WARN_PCT:
        tags.append("LIVE_SENTINEL_FLOATING_WARN")
        logger.warning(
            "LIVE_SENTINEL floating warn | dd=%.2f%% trigger=%.2f%% equity=%.2f",
            state.floating_dd_pct,
            floating_trigger_pct,
            equity,
        )

    return SentinelVerdict(
        entry_allowed=True,
        panic_close=False,
        rollover_block=False,
        spread_block=False,
        entry_locked=False,
        floating_dd_pct=state.floating_dd_pct,
        daily_dd_remaining_pct=state.daily_dd_remaining_pct,
        message=(
            f"LIVE_SENTINEL OK | floating={state.floating_dd_pct:.2f}% "
            f"remaining={state.daily_dd_remaining_pct:.2f}%"
        ),
        tags=tuple(tags),
    )


def sentinel_hold_signal(message: str, *, tags: tuple[str, ...] = ()) -> dict[str, Any]:
    """MT5 向け HOLD レスポンス。"""
    return {
        "action": "HOLD",
        "lot_size": 0.0,
        "risk_budget": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "message": message,
        "decision_source": "LIVE_SENTINEL",
        "sentinel_tags": list(tags),
    }


def sentinel_panic_signal(message: str, *, tags: tuple[str, ...] = ()) -> dict[str, Any]:
    """MT5 向け PANIC_CLOSE レスポンス — EA が全決済・全キャンセル実行。"""
    return {
        "action": "PANIC_CLOSE",
        "lot_size": 0.0,
        "risk_budget": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "message": message,
        "decision_source": "LIVE_SENTINEL",
        "sentinel_tags": list(tags),
    }
