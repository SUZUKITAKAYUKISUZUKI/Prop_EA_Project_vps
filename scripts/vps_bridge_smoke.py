"""
VPS bridge smoke test — run on the VPS after git pull / sync.

  python scripts/vps_bridge_smoke.py
  py -3 scripts/vps_bridge_smoke.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "deploy" / "vps-min-manifest.json"


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _manifest_version() -> str:
    if not MANIFEST_PATH.is_file():
        return "unknown"
    try:
        import json as _json

        data = _json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return str(data.get("version", "?"))
    except Exception:
        return "unknown"


def main() -> None:
    print("=== Prop EA VPS bridge smoke test ===")
    print(f"Root: {ROOT}")
    print(f"Manifest: deploy/vps-min-manifest.json v{_manifest_version()}")

    required = [
        ROOT / "strategies" / "bt_ohlcv.py",
        ROOT / "strategies" / "london_sweep_failure.py",
        ROOT / "strategies" / "dbbs.py",
        ROOT / "strategies" / "dbbs_common.py",
        ROOT / "strategies" / "dbbs_exit.py",
        ROOT / "strategies" / "dbbs_bear_kill_switch.py",
        ROOT / "strategies" / "scan_numba_util.py",
        ROOT / "strategies" / "dinapoli.py",
        ROOT / "strategies" / "dinapoli_mtf.py",
        ROOT / "strategies" / "dinapoli_universe_fast.py",
        ROOT / "src" / "filters" / "dn_prop_gate_runtime.py",
        ROOT / "backtest_results" / "models" / "dn_bayes_ev_v2.json",
        ROOT / "backtest_results" / "models" / "dn_prop_gate_v1.json",
        ROOT / "storage" / "dn_feature_store.py",
        ROOT / "strategies" / "vamr.py",
        ROOT / "strategies" / "vamr_bayes.py",
        ROOT / "strategies" / "vamr_features.py",
        ROOT / "strategies" / "vamr_phase2.py",
        ROOT / "strategies" / "var_reversal.py",
        ROOT / "strategies" / "var_detector.py",
        ROOT / "backtest_results" / "models" / "vamr_bayes_v1.json",
        ROOT / "strategies" / "smrs_pure.py",
        ROOT / "strategies" / "smrs.py",
        ROOT / "strategies" / "smrs_scan_numba.py",
        ROOT / "strategies" / "smrs_bayes.py",
        ROOT / "strategies" / "smrs_sizing.py",
        ROOT / "strategies" / "smrs_production.py",
        ROOT / "backtest_results" / "models" / "smrs_bayes_v1.json",
        ROOT / "audit" / "live_tp_cap.py",
        ROOT / "mt5" / "DbbsExitManager.mqh",
        ROOT / "mt5" / "PropEA_Bridge.mq5",
        ROOT / "deploy" / "portfolio_allocation_weights.json",
    ]
    excluded_bt_only = [
        ROOT / "strategies" / "smrs_portfolio.py",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        _fail(
            "Missing VPS minimum files (manifest v12 / A+B+C+D+E):\n  "
            + "\n  ".join(str(p.relative_to(ROOT)) for p in missing)
            + "\nRe-sync from dev (sync_vps_min.cmd) or git pull the latest VPS repo."
        )
    present_bt_only = [p for p in excluded_bt_only if p.is_file()]
    if present_bt_only:
        print(
            "[WARN] BT-only SMRS modules present (not required on VPS): "
            + ", ".join(p.name for p in present_bt_only)
        )
    print("[OK] VPS minimum files present (A+B+C+D+E)")

    try:
        from strategies import STRATEGY_LETTER_BY_MODE, STRATEGY_LETTER_BY_SETUP_TYPE, expand_strategy_modes

        assert expand_strategy_modes("abcde") == ("lsfc", "dbbs", "dinapoli", "vamr", "smrs")
        assert STRATEGY_LETTER_BY_MODE["smrs"] == "E"
        from strategies import get_live_strategies

        live_types = {s.setup_type for s in get_live_strategies({}, mode_h1=True)}
        assert "SMRS" in live_types
        from strategies.smrs_production import PRODUCTION_SPEC, configure_smrs_defense_env

        configure_smrs_defense_env()
        assert PRODUCTION_SPEC.letter == "E"
        assert PRODUCTION_SPEC.bayes_enabled is True
        assert PRODUCTION_SPEC.gemini_audit is False
        assert PRODUCTION_SPEC.pyramiding is False
        from strategies.smrs import SmrsStrategy

        assert SmrsStrategy({}, mode_h1=False).setup_type == "SMRS"
    except Exception as exc:
        print("[FAIL] strategy registry check:")
        traceback.print_exc()
        _fail(str(exc))
    print("[OK] Strategy registry (letter E / abcde expansion)")

    try:
        from audit.risk_manager import (
            finalize_lot_factor_for_execution,
            is_fintokei_single_position_rule_enabled,
        )
        from strategies.dbbs_common import DBBS_MAX_LOSS_R
        from strategies.dbbs_exit import build_dbbs_exit_signal_fields, is_dbbs_live_trail_enabled

        assert DBBS_MAX_LOSS_R == 1.0
        if is_dbbs_live_trail_enabled():
            fields = build_dbbs_exit_signal_fields()
            assert fields.get("exit_mode") == "DBBS_TRAIL"
            assert fields.get("exit_max_loss_r") == 1.0
        lf, _rb, _ls, _tags, reject = finalize_lot_factor_for_execution(
            6.0,
            base_risk_pct=0.006,
            sl_distance=0.0020,
            equity=100_000.0,
            daily_committed_risk_pct=0.0,
            max_loss_r=1.0,
        )
        assert reject is False
        assert lf > 0.0
        if is_fintokei_single_position_rule_enabled():
            assert lf <= 5.0
    except Exception as exc:
        print("[FAIL] live exit / lot_factor cap check:")
        traceback.print_exc()
        _fail(str(exc))
    print("[OK] Live phase2 exits + Fintokei 3% lot cap")

    try:
        import pandas as pd
        from feature_engineering import LivePipelineState, evaluate_trade_signal
    except Exception as exc:
        print("[FAIL] import error:")
        traceback.print_exc()
        _fail(str(exc))

    start = pd.Timestamp("2026-05-28 00:00:00")
    bars = [
        {
            "time": (start + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 1.27,
            "high": 1.271,
            "low": 1.269,
            "close": 1.270,
            "volume": 100.0,
        }
        for i in range(300)
    ]
    payload = {
        "market": {
            "pair": "GBPUSD",
            "open": 1.27,
            "high": 1.271,
            "low": 1.269,
            "close": 1.270,
            "volume": 100.0,
        },
        "calendar": {"minutes_to_next_news": 45, "news_impact_level": "HIGH"},
        "account": {"equity": 100000.0, "balance": 100000.0},
        "bar_time": bars[-1]["time"],
        "server_time": bars[-1]["time"],
        "spread_points": 10,
        "bars": bars,
        "correlated_market": {
            "pair": "EURUSD",
            "open": 1.08,
            "high": 1.081,
            "low": 1.079,
            "close": 1.080,
            "volume": 100.0,
        },
        "correlated_bar_time": bars[-1]["time"],
        "correlated_bars": bars,
    }

    try:
        result = evaluate_trade_signal(payload, LivePipelineState.create())
    except Exception as exc:
        print("[FAIL] evaluate_trade_signal raised:")
        traceback.print_exc()
        _fail(str(exc))

    print("[OK] evaluate_trade_signal")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:800])
    print("\nIf this passes but MT5 still returns 500, check the bridge window for")
    print("'POST /trade_signal failed' and paste the traceback here.")


if __name__ == "__main__":
    main()
