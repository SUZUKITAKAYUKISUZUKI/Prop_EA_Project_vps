# archive/lgr

Liquidity Grab Reversal (LGR) および関連モジュールの保管場所。

## 退役理由（2026-06-10）

**プロップファーム向きではない**（DD・一貫性の観点）。**自己資金口座用 EA としては非常に優秀**なため、コード・結果は参照用に保持。

## 構成

| パス | 内容 |
|---|---|
| `strategies/archive/liquidity_grab_reversal.py` 等 | 戦略本体・スキャナ・特徴量ヘルパー |
| `archive/lgr/*.py` | Bayes / EV sizing / Prop Optuna 制御 |
| `archive/lgr/tests/` | 単体テスト |
| `scripts/archive/run_lgr_*.ps1` 等 | BT / WFT / Optuna スクリプト（参照用） |
| `backtest_results/archive/lgr/` | 3y BT・Optuna・WFT 結果 |

## 使い方

- 本番・通常 BT: **`lgr` は `ARCHIVED_STRATEGY_MODES` 登録済み** — `backtest_runner --strategy lgr` は拒否される
- 再分析: `from strategies.archive.liquidity_grab_reversal import ...` / `from archive.lgr.lgr_ev_position_sizing import ...`
- テスト: `pytest archive/lgr/tests/`
- Prop Optuna（参照）: `python -m archive.lgr.optimize_lgr_prop`（WFT 内 BT も archived 拒否のため、復活時は registry から外すか `--allow-archived` 相当の対応が必要）

## v2 Optuna ベスト（参考）

Trial #95 — Prop Score 10.86, PF 2.37, Sharpe 4.59, +82.0R (WFT), DD exceed 0%  
→ `backtest_results/archive/lgr/lgr_prop_optuna_best_v2.json`
