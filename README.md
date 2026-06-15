# Prop EA — VPS 最小構成リポジトリ

本フォルダは **開発リポジトリ (`Prop_EA_Project`) から同期される VPS 実運用用の最小セット** です。  
マニフェスト: **`deploy/vps-min-manifest.json` v8**（A+B+C+D+E 本番）

## 本番ポートフォリオ A+B+C+D+E

| Letter | Mode | 戦略 | Exec / 構造 / ATR | 本番ペア |
|--------|------|------|-------------------|----------|
| **A** | `lsfc` | London Sweep Failure Continuation | M15 / H1 | GBPUSD, EURUSD |
| **B** | `dbbs` | Dual Bollinger Band Squeeze + Bear Kill Switch V2 | M15 / H1 / H4 | **EURUSD, GBPUSD, XAUUSD** |
| **C** | `dinapoli` | DiNapoli Structure + DN Prop Gate V1 | M15 / H1 / H4 | ユニバース |
| **D** | `vamr` | **Volume Area Mean Reversion to POC**（略称 **VAMR**） | H1 / M5 VP / H4 | **AUDNZD, EURGBP, USDCAD** |
| **E** | `smrs` | **Statistical Mean Reversion Scalper**（略称 **SMRS**） | M1 | **AUDNZD, EURGBP, NZDUSD** |

- Python Bridge は `STRATEGY_LETTER_BY_MODE` に登録された **A/B/C/D/E** を同一エクイティで順次評価します。
- **E（Statistical Mean Reversion Scalper / 略称 SMRS）** — Phase 3 Bayes + Model A sizing。レジストリ・モデル JSON は同梱。**Live 執行**は `BaseStrategy` 実装追加まで BT merge 正本（`main_abcde_3y.csv`）。
- **AUDNZD / EURGBP** は D と E で共有 — **setup_type 単位の L2**（同一戦略×シンボル最大1）で競合を回避。
- BT / WFT / 巨大 CSV / checkpoints は **含みません**（Bayes / DN Prop Gate 用モデル JSON のみ同梱）。

同期手順: [`VPS_MIN_SYNC_GUIDE.md`](./VPS_MIN_SYNC_GUIDE.md)

## 3y BT 基準（A+B+C+D+E, allocation OFF）

| Metric | Value |
|--------|------:|
| Executed trades | 2,472 |
| Total R (effective) | +1,120.84R |
| PF | 3.127 |
| Prop pass rate | 100.00% |
| Avg pass days | 6.0 |
| WFT positive windows | 100% |

## VPS 推奨 `.env`

`.env.example` を `.env` にコピーし、`GEMINI_API_KEY` を設定してください。

```ini
GEMINI_API_KEY=your_key_here
PROFIT_CUSHION_ENABLED=1
PROFIT_CUSHION_LOT_MULT=0.65
TWIN_BRAKE_ENABLED=1
DD_THROTTLING_ENABLED=1
MUTUAL_EXCLUSION_ENABLED=1
PYRAMID_ENABLED=1
PYRAMID_LSFC=1
PORTFOLIO_ALLOCATION_ENABLED=0
DBBS_DEFENSE=1
DBBS_BEAR_KILL_SWITCH=1
DINAPOLI_DEFENSE=1
DN_PROP_GATE=1
VAMR_DEFENSE=1
VAMR_GEMINI_AUDIT=0
SMRS_GEMINI_AUDIT=0
SMRS_LLM_AUDIT=0
```

**Profit Cushion ×0.65** は **A/B/C/D/E 全戦略共通** の L4.5 防御です。

## Portfolio Strategy Allocation（任意 — 本番 OFF）

| 変数 | 説明 |
|------|------|
| `PORTFOLIO_ALLOCATION_ENABLED=0` | **本番既定** — lot_factor にウェイトを掛けない |
| `PORTFOLIO_ALLOCATION_ENABLED=1` | 最適化ウェイトを lot_factor に反映 |
| `PORTFOLIO_STRATEGY_WEIGHTS_FILE` | `deploy/portfolio_allocation_weights.json`（参考値。再最適化推奨） |

## L2 — 戦略×シンボル 1 ポジション + ピラミッディング

| 層 | 動作 |
|---|---|
| **Python L2** | `one_per_strategy_symbol` — A/B/C/D/E 各戦略でシンボルごとに最大1 |
| **ピラミッディング** | 有効戦略は L2 **自動 OFF**（既定: LSFC のみ ON） |
| **MT5 EA** | `open_positions[]` に `setup_type` / `strategy_letter` を送信 |
| **Bridge JSON** | comment `PropEA_A` … `PropEA_E` → setup_type へマップ |

`MUTUAL_EXCLUSION_MODE=daily` / `concurrent` は **廃止**。VPS `.env` から削除してください。

**VPS 反映時:** `PropEA_Bridge.mq5` を **再コンパイル・再アタッチ**（letter E 対応版）。

## 同梱ファイル（manifest v8 必須）

**A — LSFC**

- `strategies/london_sweep_failure.py` / `lsfc_scan_hot.py` / `lsfc_scan_numba.py`

**B — DBBS**

- `strategies/dbbs.py` / `dbbs_common.py` / `dbbs_bear_kill_switch.py` / `scan_numba_util.py`

**C — DiNapoli**

- `strategies/dinapoli.py` / `dinapoli_mtf.py` / `dinapoli_feature_log.py` / `dinapoli_universe_fast.py`
- `src/filters/dn_prop_gate_*.py` / `dn_bayes_ev_v2.py`
- `backtest_results/models/dn_bayes_ev_v2.json` / `dn_prop_gate_v1.json`
- `storage/dn_feature_store.py`

**D — Volume Area Mean Reversion to POC（略称 VAMR）**

- `strategies/vamr.py` / `vamr_bayes.py` / `vamr_features.py` / `vamr_phase2.py`
- `strategies/var_reversal.py` / `var_detector.py`
- `backtest_results/models/vamr_bayes_v1.json`

**E — Statistical Mean Reversion Scalper（略称 SMRS）**

- `strategies/smrs_pure.py` / `smrs_scan_numba.py` / `smrs_bayes.py` / `smrs_production.py`
- `backtest_results/models/smrs_bayes_v1.json`
- **除外（BT のみ）:** `smrs_portfolio.py` / `smrs_sizing.py`

## VPS 反映チェックリスト

1. 開発 PC: `scripts\sync_vps_min.cmd` → manifest **v8** を確認
2. VPS: `git pull` → `py -3 scripts\vps_bridge_smoke.py` が `[OK]`
3. `.env` を `.env.example` どおりに設定（`PORTFOLIO_ALLOCATION_ENABLED=0`）
4. MT5: `PropEA_Bridge.mq5` 再コンパイル（letter **E** 対応）
5. `start_mt5_bridge.bat` 再起動

`.gitignore` は `storage/` 配下の実行時 DB を除外しますが、上記 Python モジュールは同期対象です。
