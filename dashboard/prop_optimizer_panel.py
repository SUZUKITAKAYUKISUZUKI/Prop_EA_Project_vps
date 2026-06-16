"""Phase 5.10 — Prop Optimizer dashboard panel (FastAPI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from audit.risk_manager import STARTING_EQUITY
from core.pass_probability import AccountSnapshot, ChallengeState
from core.prop_optimizer import PropOptimizer, run_pfoo
from core.prop_profiles import load_pfoo_config, load_prop_profiles

router = APIRouter(prefix="/dashboard/prop_optimizer", tags=["pfoo"])

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATE: dict[str, Any] = {
    "last_result": None,
    "account": {
        "equity": STARTING_EQUITY,
        "balance": STARTING_EQUITY,
        "peak_equity": STARTING_EQUITY,
    },
    "challenge": {
        "days_elapsed": 0,
        "profit_progress_percent": 0.0,
        "daily_dd_used_percent": 0.0,
        "total_dd_used_percent": 0.0,
    },
    "mode": "BALANCED",
    "profile": "Fintokei_100K",
}


def _snapshot_from_state() -> tuple[AccountSnapshot, ChallengeState]:
    acct_raw = _STATE["account"]
    ch_raw = _STATE["challenge"]
    account = AccountSnapshot(
        equity=float(acct_raw["equity"]),
        balance=float(acct_raw["balance"]),
        peak_equity=float(acct_raw["peak_equity"]),
    )
    challenge = ChallengeState(
        days_elapsed=int(ch_raw["days_elapsed"]),
        profit_progress_percent=float(ch_raw["profit_progress_percent"]),
        daily_dd_used_percent=float(ch_raw["daily_dd_used_percent"]),
        total_dd_used_percent=float(ch_raw["total_dd_used_percent"]),
    )
    return account, challenge


def get_panel_state() -> dict[str, Any]:
    result = _STATE.get("last_result")
    if result is None:
        return {
            "loaded": False,
            "mode": _STATE["mode"],
            "profile": _STATE["profile"],
            "account": _STATE["account"],
            "challenge": _STATE["challenge"],
            "operational_status": "NORMAL",
        }
    return {
        "loaded": True,
        "mode": result.mode,
        "profile": result.profile.name,
        "operational_status": result.operational_status,
        "pass_probability": result.pass_probability.pass_probability,
        "expected_pass_days": result.expected_pass_days,
        "utility": result.utility.utility,
        "progress_risk_multiplier": result.progress_risk_multiplier,
        "recommended_weights": result.recommended_weights,
        "monte_carlo": {
            str(k): {
                "pass_rate": v.pass_rate,
                "fail_rate": v.fail_rate,
                "avg_pass_days": v.avg_pass_days,
                "expected_utility": v.expected_utility,
            }
            for k, v in result.monte_carlo.items()
        },
        "account": _STATE["account"],
        "challenge": _STATE["challenge"],
        "recovery_active": result.recovery.active,
        "endgame_active": result.endgame.active,
    }


@router.get("", response_class=HTMLResponse)
async def prop_optimizer_dashboard() -> str:
    state = get_panel_state()
    weights_json = json.dumps(state.get("recommended_weights") or {}, indent=2)
    mc_rows = ""
    for trials, mc in (state.get("monte_carlo") or {}).items():
        mc_rows += f"<tr><td>{trials}</td><td>{mc['pass_rate']:.1f}%</td><td>{mc['avg_pass_days']:.1f}</td><td>{mc['expected_utility']:.3f}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>PFOO — Prop Optimizer</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #0f1419; color: #e6edf3; }}
    h1 {{ color: #58a6ff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
    .gauge {{ font-size: 2rem; font-weight: 700; color: #3fb950; }}
    .status {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #238636; }}
    .status.recovery {{ background: #9e6a03; }}
    .status.endgame {{ background: #8957e5; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #30363d; padding: 8px; text-align: right; }}
    pre {{ background: #0d1117; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Prop Firm Objective Optimizer</h1>
  <p>Profile: <strong>{state.get('profile', '—')}</strong> | Mode: <strong>{state.get('mode', 'BALANCED')}</strong>
     | Status: <span class="status {'recovery' if state.get('operational_status')=='RECOVERY' else 'endgame' if state.get('operational_status')=='ENDGAME' else ''}">{state.get('operational_status', 'NORMAL')}</span></p>

  <div class="grid">
    <div class="card">
      <h3>Challenge Progress</h3>
      <div class="gauge">{state.get('challenge', {}).get('profit_progress_percent', 0):.1f}%</div>
      <div>Remaining target: {max(0, 8 - state.get('challenge', {}).get('profit_progress_percent', 0)):.1f}%</div>
      <div>DD used: {state.get('challenge', {}).get('total_dd_used_percent', 0):.1f}%</div>
    </div>
    <div class="card">
      <h3>Pass Probability</h3>
      <div class="gauge">{state.get('pass_probability', 0):.1f}%</div>
    </div>
    <div class="card">
      <h3>Expected Pass Days</h3>
      <div class="gauge">{state.get('expected_pass_days', 0):.1f}</div>
    </div>
    <div class="card">
      <h3>Utility</h3>
      <div class="gauge">{state.get('utility', 0):.3f}</div>
      <div>Risk mult: {state.get('progress_risk_multiplier', 1):.2f}</div>
    </div>
  </div>

  <h2>Risk Budget Allocation</h2>
  <pre>{weights_json}</pre>

  <h2>Monte Carlo Projection</h2>
  <table>
    <tr><th>Trials</th><th>Pass Rate</th><th>Avg Pass Days</th><th>Utility</th></tr>
    {mc_rows or '<tr><td colspan="4">Run POST /dashboard/prop_optimizer/refresh to load</td></tr>'}
  </table>
</body>
</html>"""


@router.get("/state")
async def prop_optimizer_state() -> dict[str, Any]:
    return get_panel_state()


@router.post("/refresh")
async def prop_optimizer_refresh(
    mode: str = "BALANCED",
    profile: str | None = None,
) -> dict[str, Any]:
    _STATE["mode"] = mode.upper()
    if profile:
        _STATE["profile"] = profile
    account, challenge = _snapshot_from_state()
    result = run_pfoo(
        profile_name=_STATE["profile"],
        mode=_STATE["mode"],  # type: ignore[arg-type]
        account=account,
        challenge=challenge,
    )
    _STATE["last_result"] = result
    return get_panel_state()


@router.post("/update_account")
async def update_account(
    equity: float,
    balance: float | None = None,
    peak_equity: float | None = None,
    profit_progress_percent: float | None = None,
    total_dd_used_percent: float | None = None,
    days_elapsed: int | None = None,
) -> dict[str, Any]:
    _STATE["account"]["equity"] = equity
    _STATE["account"]["balance"] = balance if balance is not None else equity
    _STATE["account"]["peak_equity"] = peak_equity if peak_equity is not None else max(equity, _STATE["account"]["peak_equity"])
    if profit_progress_percent is not None:
        _STATE["challenge"]["profit_progress_percent"] = profit_progress_percent
    if total_dd_used_percent is not None:
        _STATE["challenge"]["total_dd_used_percent"] = total_dd_used_percent
    if days_elapsed is not None:
        _STATE["challenge"]["days_elapsed"] = days_elapsed
    return get_panel_state()


@router.get("/profiles")
async def list_profiles() -> dict[str, Any]:
    profiles = load_prop_profiles()
    return {name: {"target_profit": p.target_profit, "total_dd_limit": p.total_dd_limit} for name, p in profiles.items()}


def register_dashboard(app) -> None:
    """Mount PFOO dashboard routes on a FastAPI app."""
    app.include_router(router)
