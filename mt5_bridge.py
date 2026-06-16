"""
mt5_bridge.py — FastAPI bridge between MetaTrader 5 and feature_engineering.py (v1.7)

統合ランタイム（bridge_runtime）:
  - Gemini 1.5 Flash API 監査
  - 経済カレンダー cache/calendar.json 定期更新
  - LLM 監査 (llm_auditor) 有効化

起動:
    start_mt5_bridge.bat
    uvicorn mt5_bridge:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from feature_engineering import (
    LivePipelineState,
    TIMEFRAME_LABEL,
    evaluate_trade_signal,
)
from audit.live_sentinel import evaluate_live_sentinel, is_live_sentinel_enabled, parse_server_time

logger = logging.getLogger("mt5_bridge")
_pipeline_state = LivePipelineState.create()


class MarketPayload(BaseModel):
    pair: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class CalendarPayload(BaseModel):
    minutes_to_next_news: int = Field(default=45, ge=0)
    news_impact_level: str = "LOW"


class AccountPayload(BaseModel):
    equity: float
    balance: float


class OpenPositionPayload(BaseModel):
    pair: str
    entry_time: str | None = None
    setup_type: str | None = None


class BarPayload(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class TradeSignalRequest(BaseModel):
    market: MarketPayload
    calendar: CalendarPayload = Field(default_factory=CalendarPayload)
    account: AccountPayload
    bar_time: str | None = None
    server_time: str | None = None
    spread_points: int | None = Field(default=None, ge=0)
    bars: list[BarPayload] | None = None
    correlated_market: MarketPayload | None = None
    correlated_bar_time: str | None = None
    correlated_bars: list[BarPayload] | None = None
    open_positions: list[OpenPositionPayload] | None = None


class SentinelTickRequest(BaseModel):
    server_time: str
    equity: float
    balance: float
    spread_points: int | None = Field(default=None, ge=0)


class SentinelStatusResponse(BaseModel):
    enabled: bool
    entry_allowed: bool
    panic_close: bool
    rollover_block: bool
    spread_block: bool
    entry_locked: bool
    floating_dd_pct: float
    daily_dd_remaining_pct: float
    message: str
    tags: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


class TradeSignalResponse(BaseModel):
    action: str
    lot_size: float
    risk_budget: float = 0.0
    sl: float
    tp: float
    message: str
    entry: float | None = None
    decision_source: str | None = None
    trade_id: str | None = None
    lot_factor: float | None = None
    risk_score: int | None = None
    multipliers: dict[str, float] | None = None
    sentinel_tags: list[str] | None = None
    setup_type: str | None = None
    exit_mode: str | None = None
    exit_atr: float | None = None
    exit_be_enabled: int | None = None
    exit_trail_enabled: int | None = None
    exit_be_arm_mfe_r: float | None = None
    exit_be_trigger_mfe_r: float | None = None
    exit_be_pullback_close_r: float | None = None
    exit_be_rhythm_max_bars: int | None = None
    exit_trail_atr_mult: float | None = None
    exit_be_buffer_atr: float | None = None


class PyramidRegisterRequest(BaseModel):
    trade_id: str
    setup_type: str
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    atr: float
    lot_size: float
    base_ticket: int
    entry_bar_index: int = 0
    daily_dd_remaining_pct: float = Field(default=5.0, ge=0.0)
    kalman_velocity_at_entry: float = 0.0
    ws_mode: bool = False
    pyramid_group_id: str | None = None
    tick_size: float = Field(default=0.0, ge=0.0)
    tick_value: float = Field(default=0.0, ge=0.0)


class PyramidTickRequest(BaseModel):
    trade_id: str
    bar: BarPayload
    bar_index: int = Field(ge=0)
    daily_dd_remaining_pct: float | None = Field(default=None, ge=0.0)
    kalman_velocity_atr: float | None = None
    kalman_velocity_min: float | None = None
    past_time_limit: bool = False
    decel_exit: bool = False


class PyramidFillRequest(BaseModel):
    trade_id: str
    fill_price: float
    position_ticket: int | None = None
    order_ticket: int | None = None


class PyramidCloseRequest(BaseModel):
    trade_id: str


class PyramidActionResponse(BaseModel):
    actions: list[dict[str, Any]] = Field(default_factory=list)
    pyramid_group_id: str | None = None
    pyramid_layers: int = 0
    pending_limit: bool = False
    ws_pyramid_rejected_reason: str = ""


class PyramidRegisterResponse(BaseModel):
    trade_id: str
    pyramid_group_id: str
    live_pyramid_enabled: bool
    message: str = "registered"


class DbbsTradeClosedRequest(BaseModel):
    result_r: float


class DbbsTradeClosedResponse(BaseModel):
    ok: bool = True
    last_3_avg_r: float | None = None
    bear_kill_switch_active: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    from bridge_runtime import shutdown_bridge_runtime, startup_bridge_runtime

    runtime_summary = await asyncio.to_thread(startup_bridge_runtime)
    logger.info("Bridge runtime started: %s", runtime_summary)

    yield

    await asyncio.to_thread(shutdown_bridge_runtime)


app = FastAPI(
    title="Prop EA MT5 Bridge",
    version="1.1.0",
    description="7-layer pipeline HTTP bridge for MetaTrader 5 (Gemini + Calendar + LLM)",
    lifespan=lifespan,
)

try:
    from dashboard.prop_optimizer_panel import register_dashboard as register_pfoo_dashboard

    register_pfoo_dashboard(app)
except ImportError:
    logger.warning("PFOO dashboard not available (dashboard package missing)")

try:
    from dashboard.pet_panel import register_dashboard as register_pet_dashboard

    register_pet_dashboard(app)
except ImportError:
    logger.warning("PET dashboard not available (dashboard package missing)")


def _request_to_dict(request: TradeSignalRequest) -> dict[str, Any]:
    payload = request.model_dump(exclude_none=True)
    payload["market"] = request.market.model_dump()
    payload["calendar"] = request.calendar.model_dump()
    payload["account"] = request.account.model_dump()
    if request.bars:
        payload["bars"] = [b.model_dump() for b in request.bars]
    if request.correlated_market:
        payload["correlated_market"] = request.correlated_market.model_dump()
    if request.correlated_bars:
        payload["correlated_bars"] = [b.model_dump() for b in request.correlated_bars]
    if request.open_positions:
        payload["open_positions"] = [p.model_dump(exclude_none=True) for p in request.open_positions]
    return payload


@app.get("/health")
async def health() -> dict[str, str]:
    from bridge_runtime import get_runtime_status

    status = get_runtime_status()
    return {
        "status": "ok",
        "pipeline_mode": TIMEFRAME_LABEL,
        **status,
    }


@app.post("/trade_signal", response_model=TradeSignalResponse)
async def trade_signal(request: TradeSignalRequest) -> TradeSignalResponse:
    global _pipeline_state
    try:
        result = evaluate_trade_signal(_request_to_dict(request), _pipeline_state)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc}") from exc
    except Exception as exc:
        logger.exception("POST /trade_signal failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        from dashboard.pet_panel import update_pet_from_decision

        if _pipeline_state.last_pet_decision is not None:
            update_pet_from_decision(_pipeline_state.last_pet_decision, _pipeline_state.pet)
    except ImportError:
        pass
    return TradeSignalResponse(**result)


@app.post("/dbbs/trade_closed", response_model=DbbsTradeClosedResponse)
async def dbbs_trade_closed(request: DbbsTradeClosedRequest) -> DbbsTradeClosedResponse:
    """DBBS Bear Kill Switch V2 — update edge memory after a closed trade."""
    from strategies.dbbs_bear_kill_switch import get_edge_tracker, record_closed_trade_result

    record_closed_trade_result(request.result_r)
    snap = get_edge_tracker().pre_trade_snapshot()
    last3 = snap.get("last_3_avg_r")
    last3_out = float(last3) if last3 is not None and str(last3) != "nan" else None
    return DbbsTradeClosedResponse(
        ok=True,
        last_3_avg_r=last3_out,
        bear_kill_switch_active=bool(snap.get("bear_kill_switch_active")),
    )


@app.post("/sentinel/tick", response_model=SentinelStatusResponse)
async def sentinel_tick(request: SentinelTickRequest) -> SentinelStatusResponse:
    """OnTick 高頻度監視 — EA から server_time / equity を送信。"""
    global _pipeline_state
    server_dt = parse_server_time(request.server_time)
    verdict = evaluate_live_sentinel(
        _pipeline_state.sentinel,
        server_dt,
        request.balance,
        request.equity,
        spread_points=request.spread_points,
        enabled=is_live_sentinel_enabled(),
    )
    if verdict.log_level == "error":
        logger.error(verdict.message)
    elif verdict.log_level == "warning":
        logger.warning(verdict.message)
    return SentinelStatusResponse(
        enabled=is_live_sentinel_enabled(),
        entry_allowed=verdict.entry_allowed,
        panic_close=verdict.panic_close,
        rollover_block=verdict.rollover_block,
        spread_block=verdict.spread_block,
        entry_locked=verdict.entry_locked,
        floating_dd_pct=verdict.floating_dd_pct,
        daily_dd_remaining_pct=verdict.daily_dd_remaining_pct,
        message=verdict.message,
        tags=list(verdict.tags),
        state=_pipeline_state.sentinel.to_dict(),
    )


@app.get("/sentinel/status", response_model=dict[str, Any])
async def sentinel_status() -> dict[str, Any]:
    """現在の Sentinel 状態スナップショット。"""
    return {
        "enabled": is_live_sentinel_enabled(),
        "state": _pipeline_state.sentinel.to_dict(),
    }


@app.get("/pet/status", response_model=dict[str, Any])
async def pet_status() -> dict[str, Any]:
    """Portfolio Equity Trail runtime snapshot."""
    from core.portfolio_equity_trail import is_pet_enabled

    decision = _pipeline_state.last_pet_decision
    runtime = _pipeline_state.pet
    payload: dict[str, Any] = {
        "enabled": is_pet_enabled(_pipeline_state.account.profile),
        "profile": _pipeline_state.account.profile,
        "runtime": None,
        "decision": None,
    }
    if runtime is not None:
        payload["runtime"] = {
            "day_start_equity": runtime.day_start_equity,
            "peak_equity": runtime.peak_equity,
            "protected_equity": runtime.protected_equity,
            "peak_gain_r": runtime.peak_gain_r,
            "locked_profit_r": runtime.locked_profit_r,
            "stage_name": runtime.stage_name,
            "active": runtime.active,
            "breached": runtime.breached,
            "disable_new_entries": runtime.disable_new_entries,
            "trading_halted_for_day": runtime.trading_halted_for_day,
            "last_action": runtime.last_action,
            "server_day": runtime.server_day,
        }
    if decision is not None:
        payload["decision"] = decision.to_dict()
    return payload


@app.post("/reset_state")
async def reset_state() -> dict[str, str]:
    """バックテスト/デモ切替時にセッション状態をリセット。"""
    global _pipeline_state
    _pipeline_state = LivePipelineState.create()
    from bridge_runtime import get_live_pyramid_registry

    get_live_pyramid_registry().reset()
    return {"status": "reset"}


def _pyramid_action_response(session, actions) -> PyramidActionResponse:
    return PyramidActionResponse(
        actions=[a.to_dict() for a in actions],
        pyramid_group_id=session.pyramid_group_id,
        pyramid_layers=session.mgr.pyramid_layers,
        pending_limit=session.pending_limit is not None,
        ws_pyramid_rejected_reason=session.last_rejected_reason,
    )


@app.post("/pyramid/register", response_model=PyramidRegisterResponse)
async def pyramid_register(request: PyramidRegisterRequest) -> PyramidRegisterResponse:
    """初回成行約定後 — Live Pyramid セッションを registry に登録。"""
    from bridge_runtime import get_live_pyramid_registry
    from live_pyramid.config import is_live_pyramid_enabled

    if not is_live_pyramid_enabled(request.setup_type):
        raise HTTPException(
            status_code=409,
            detail=f"live pyramid disabled for setup_type={request.setup_type}",
        )
    registry = get_live_pyramid_registry()
    try:
        session = registry.create_and_register(
            trade_id=request.trade_id,
            setup_type=request.setup_type,
            symbol=request.symbol,
            direction=request.direction,
            entry=request.entry,
            sl=request.sl,
            tp=request.tp,
            atr=request.atr,
            initial_lot=request.lot_size,
            base_ticket=request.base_ticket,
            entry_bar_index=request.entry_bar_index,
            daily_dd_remaining_percent=request.daily_dd_remaining_pct,
            kalman_velocity_at_entry=request.kalman_velocity_at_entry,
            ws_mode=request.ws_mode,
            pyramid_group_id=request.pyramid_group_id,
            tick_size=request.tick_size,
            tick_value=request.tick_value,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    from live_pyramid.l6_log import log_pyramid_register

    log_pyramid_register(session)
    return PyramidRegisterResponse(
        trade_id=session.trade_id,
        pyramid_group_id=session.pyramid_group_id,
        live_pyramid_enabled=True,
    )


@app.post("/pyramid/tick", response_model=PyramidActionResponse)
async def pyramid_tick(request: PyramidTickRequest) -> PyramidActionResponse:
    """確定バーごとのピラミッド評価 — Limit / SL 更新指令を返す。"""
    from bridge_runtime import get_live_pyramid_registry
    from live_pyramid.evaluator import BarSnapshot
    from live_pyramid.session import WyckoffGateInput

    registry = get_live_pyramid_registry()
    session = registry.get_by_trade_id(request.trade_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {request.trade_id}")

    ws_gates = None
    if session.ws_mode or request.past_time_limit:
        ws_gates = WyckoffGateInput(
            past_time_limit=request.past_time_limit,
        )

    bar = BarSnapshot(
        open=request.bar.open,
        high=request.bar.high,
        low=request.bar.low,
        close=request.bar.close,
    )
    actions = registry.evaluate_tick(
        request.trade_id,
        bar,
        bar_index=request.bar_index,
        daily_dd_remaining=request.daily_dd_remaining_pct,
        ws_gates=ws_gates,
    )
    from live_pyramid.l6_log import log_pyramid_tick

    log_pyramid_tick(session, actions, bar_index=request.bar_index)
    return _pyramid_action_response(session, actions)


@app.post("/pyramid/fill", response_model=PyramidActionResponse)
async def pyramid_fill(request: PyramidFillRequest) -> PyramidActionResponse:
    """Limit 約定通知 — PyramidManager 状態同期 + SL 一括更新指令。"""
    from bridge_runtime import get_live_pyramid_registry

    registry = get_live_pyramid_registry()
    session = registry.get_by_trade_id(request.trade_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {request.trade_id}")

    if request.order_ticket is not None and session.pending_limit is not None:
        session.pending_limit.order_ticket = request.order_ticket

    actions = registry.notify_fill(
        request.trade_id,
        request.fill_price,
        position_ticket=request.position_ticket,
    )
    from live_pyramid.l6_log import log_pyramid_fill

    log_pyramid_fill(
        session,
        actions,
        fill_price=request.fill_price,
        position_ticket=request.position_ticket,
        order_ticket=request.order_ticket,
    )
    return _pyramid_action_response(session, actions)


@app.post("/pyramid/close", response_model=PyramidActionResponse)
async def pyramid_close(request: PyramidCloseRequest) -> PyramidActionResponse:
    """トレード終了 — 未約定 Limit キャンセル + セッション破棄。"""
    from bridge_runtime import get_live_pyramid_registry

    registry = get_live_pyramid_registry()
    session = registry.get_by_trade_id(request.trade_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session not found: {request.trade_id}")

    actions = registry.close(request.trade_id)
    from live_pyramid.l6_log import log_pyramid_close

    log_pyramid_close(session, actions)
    return PyramidActionResponse(
        actions=[a.to_dict() for a in actions],
        pyramid_group_id=session.pyramid_group_id,
        pyramid_layers=session.mgr.pyramid_layers,
        pending_limit=False,
        ws_pyramid_rejected_reason=session.last_rejected_reason,
    )


@app.get("/pyramid/config")
async def pyramid_config() -> dict[str, Any]:
    """ストラテジー別ピラミッド既定 / 環境変数 / 実効状態。"""
    from live_pyramid.config import live_pyramid_env_enabled, live_pyramid_strategy_status
    from pyramid_manager import _env_flag

    global_off = _env_flag("PYRAMID_ENABLED") is False
    return {
        "pyramid_global_enabled": not global_off,
        "live_pyramid_master_enabled": live_pyramid_env_enabled(),
        "strategies": live_pyramid_strategy_status(),
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run("mt5_bridge:app", host="127.0.0.1", port=8000, reload=True)
