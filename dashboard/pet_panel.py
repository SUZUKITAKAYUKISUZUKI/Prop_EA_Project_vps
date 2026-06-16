"""Phase 5.2 — Portfolio Equity Trail (PET) dashboard panel."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from audit.risk_manager import STARTING_EQUITY
from core.portfolio_equity_trail import load_pet_config, resolve_r_unit_usd

router = APIRouter(prefix="/dashboard/pet", tags=["pet"])

_STATE: dict[str, Any] = {
    "loaded": False,
    "current_equity": STARTING_EQUITY,
    "peak_equity": STARTING_EQUITY,
    "protected_equity": STARTING_EQUITY,
    "locked_profit_r": 0.0,
    "equity_gain_r": 0.0,
    "peak_gain_r": 0.0,
    "stage_name": "OFF",
    "mode": "SOFT",
    "status": "OFF",
    "active": False,
    "breached": False,
    "disable_new_entries": False,
    "endgame_active": False,
    "action": "NONE",
    "message": "PET awaiting live data",
}


def update_pet_from_decision(decision: Any, runtime: Any | None = None) -> None:
    """Called from mt5_bridge after each trade_signal evaluation."""
    _STATE["loaded"] = True
    _STATE["current_equity"] = decision.current_equity
    _STATE["peak_equity"] = decision.peak_equity
    _STATE["protected_equity"] = decision.protected_equity
    _STATE["locked_profit_r"] = decision.locked_profit_r
    _STATE["equity_gain_r"] = decision.equity_gain_r
    _STATE["peak_gain_r"] = decision.peak_gain_r
    _STATE["stage_name"] = decision.stage_name
    _STATE["mode"] = decision.mode
    _STATE["active"] = decision.active
    _STATE["breached"] = decision.breached
    _STATE["disable_new_entries"] = decision.disable_new_entries
    _STATE["endgame_active"] = decision.endgame.active
    _STATE["action"] = decision.action
    _STATE["message"] = decision.message
    if decision.active and not decision.breached:
        _STATE["status"] = decision.stage_name
    elif decision.breached:
        _STATE["status"] = "BREACH"
    else:
        _STATE["status"] = "OFF"
    if runtime is not None:
        _STATE["day_start_equity"] = runtime.day_start_equity
        _STATE["trading_halted_for_day"] = runtime.trading_halted_for_day


def get_panel_state() -> dict[str, Any]:
    cfg = load_pet_config()
    day_start = float(_STATE.get("day_start_equity", STARTING_EQUITY))
    r_unit = resolve_r_unit_usd(day_start, cfg)
    return {
        **_STATE,
        "r_unit_usd": r_unit,
        "config_enabled": bool(cfg.get("enabled", True)),
        "profile_defaults": cfg.get("profile_defaults") or {},
        "pet_start_r": float(cfg.get("pet_start_r", 3.0)),
    }


@router.get("", response_class=HTMLResponse)
async def pet_dashboard() -> str:
    state = get_panel_state()
    status_class = "breach" if state.get("breached") else ("active" if state.get("active") else "off")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>PET — Portfolio Equity Trail</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #0f1419; color: #e6edf3; }}
    h1 {{ color: #58a6ff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
    .metric {{ font-size: 1.6rem; font-weight: 700; color: #3fb950; }}
    .metric.warn {{ color: #d29922; }}
    .metric.danger {{ color: #f85149; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #238636; }}
    .badge.off {{ background: #484f58; }}
    .badge.breach {{ background: #da3633; }}
    .badge.endgame {{ background: #8957e5; }}
    pre {{ background: #0d1117; padding: 12px; overflow: auto; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>Portfolio Equity Trail</h1>
  <p>
    <span class="badge {status_class}">{state.get("status", "OFF")}</span>
    {"<span class='badge endgame'>ENDGAME</span>" if state.get("endgame_active") else ""}
    &nbsp; Mode: <strong>{state.get("mode", "SOFT")}</strong>
  </p>
  <div class="grid">
    <div class="card">
      <div>Current Equity</div>
      <div class="metric">{state.get("equity_gain_r", 0):+.2f}R</div>
      <div>${state.get("current_equity", 0):,.2f}</div>
    </div>
    <div class="card">
      <div>Peak Equity</div>
      <div class="metric">{state.get("peak_gain_r", 0):+.2f}R</div>
      <div>${state.get("peak_equity", 0):,.2f}</div>
    </div>
    <div class="card">
      <div>Protected Equity</div>
      <div class="metric warn">{state.get("locked_profit_r", 0):+.2f}R locked</div>
      <div>${state.get("protected_equity", 0):,.2f}</div>
    </div>
    <div class="card">
      <div>PET Stage</div>
      <div class="metric">{state.get("stage_name", "OFF")}</div>
      <div>Start @ {state.get("pet_start_r", 3)}R</div>
    </div>
  </div>
  <p style="margin-top: 20px;">{state.get("message", "")}</p>
  <h2>Raw State</h2>
  <pre>{json.dumps(state, indent=2, default=str)}</pre>
  <p><a href="/dashboard/pet/json">JSON API</a></p>
</body>
</html>"""


@router.get("/json")
async def pet_json() -> dict[str, Any]:
    return get_panel_state()


def register_dashboard(app) -> None:
    """Mount PET dashboard routes on a FastAPI app."""
    app.include_router(router)
