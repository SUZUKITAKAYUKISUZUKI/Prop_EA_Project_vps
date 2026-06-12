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


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    print("=== Prop EA VPS bridge smoke test ===")
    print(f"Root: {ROOT}")

    required = [
        ROOT / "strategies" / "bt_ohlcv.py",
        ROOT / "strategies" / "london_sweep_failure.py",
        ROOT / "strategies" / "dbbs.py",
        ROOT / "strategies" / "dbbs_common.py",
        ROOT / "strategies" / "dbbs_bear_kill_switch.py",
        ROOT / "strategies" / "scan_numba_util.py",
        ROOT / "strategies" / "dinapoli.py",
        ROOT / "strategies" / "dinapoli_mtf.py",
        ROOT / "src" / "filters" / "dn_prop_gate_runtime.py",
        ROOT / "backtest_results" / "models" / "dn_bayes_ev_v2.json",
        ROOT / "backtest_results" / "models" / "dn_prop_gate_v1.json",
        ROOT / "storage" / "dn_feature_store.py",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        _fail(
            "Missing VPS minimum files:\n  "
            + "\n  ".join(str(p.relative_to(ROOT)) for p in missing)
            + "\nRe-sync from dev (sync_vps_min.cmd) or git pull the latest VPS repo."
        )
    print("[OK] VPS minimum files present (A+B+C: LSFC + DBBS + DiNapoli)")

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
