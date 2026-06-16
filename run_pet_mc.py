#!/usr/bin/env python3
"""Run Portfolio Equity Trail (PET) Monte Carlo validation — Phase 5.2.

Usage:
    python run_pet_mc.py
    python run_pet_mc.py --trials 5000
    python run_pet_mc.py --input backtest_results/main_abcde_3y.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio_equity_trail import load_pet_config, run_pet_monte_carlo_validation


def _load_trades(paths: list[Path] | None) -> pd.DataFrame:
    if not paths:
        default = PROJECT_ROOT / "backtest_results" / "main_abcde_3y.csv"
        if default.exists():
            paths = [default]
        else:
            return pd.DataFrame({"R": [0.5, -0.8, 1.0, -1.0, 1.2, -0.5]})
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame({"R": [0.5, -0.8, 1.0, -1.0]})
    merged = pd.concat(frames, ignore_index=True)
    if "R" not in merged.columns and "profit_r" in merged.columns:
        merged["R"] = merged["profit_r"]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="PET Monte Carlo validation")
    parser.add_argument("--input", type=Path, nargs="*", default=None, help="Trade CSV with R column")
    parser.add_argument("--trials", type=int, default=None, help="Override MC trial count")
    parser.add_argument("--config", type=Path, default=None, help="PET config JSON path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "pet_mc_validation.json")
    args = parser.parse_args()

    cfg = load_pet_config(args.config)
    trials = args.trials
    if trials is None:
        mc_cfg = cfg.get("monte_carlo") or {}
        trial_list = mc_cfg.get("trials") or [1000, 5000]
        trials = int(trial_list[0])
    seed = args.seed if args.seed is not None else int((cfg.get("monte_carlo") or {}).get("random_seed", 42))

    trades = _load_trades(list(args.input) if args.input else None)
    result = run_pet_monte_carlo_validation(trades, trials=trials, config=cfg, seed=seed)

    print("=== PET Monte Carlo Validation ===")
    print(f"Trials: {trials} | Seed: {seed}")
    for label in ("pet_off", "pet_on"):
        block = result[label]
        print(
            f"\n{label.upper()}: pass={block['pass_rate']}% fail={block['fail_rate']}% "
            f"RoR={block['risk_of_ruin']}% avg_pass_days={block['avg_pass_days']} worst_dd={block['worst_dd']}%"
        )
    print(f"\nSuccess criteria met: {result['success']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Report: {args.output}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
