# Prop EA — VPS 最小構成リポジトリ

本フォルダは **開発リポジトリ (`Prop_EA_Project`) から同期される VPS 実運用用の最小セット** です。

## 本番ポートフォリオ A+B+C+D

| Letter | Mode | 戦略 | Exec / 構造 / ATR |
|--------|------|------|-------------------|
| **A** | `lsfc` | London Sweep Failure Continuation | M15 / H1 |
| **B** | `dbbs` | Dual Bollinger Band Squeeze + Bear Kill Switch V2 | M15 / H1 / H4 — **EURUSD, GBPUSD, XAUUSD** |
| **C** | `dinapoli` | DiNapoli Structure + DN Prop Gate V1 | M15 / H1 / H4 |
| **D** | `vamr` | Volume Area Mean Reversion to POC (VAMR) | H1 exec / M5 VP / H4 HTF — **AUDNZD, EURGBP, USDCAD** |

- Python Bridge は `STRATEGY_LETTER_BY_MODE` に登録された **A/B/C/D の4戦略** を同一エクイティで順次評価します。
- **CSPA (旧 B)** はアーカイブ（VPS Live 対象外。`main_platform` 互換のため archive コードのみ同梱）。
- BT / WFT / 巨大 CSV / checkpoints は **含みません**（Bayes / DN Prop Gate 用モデル JSON のみ同梱）。

同期手順: [`VPS_MIN_SYNC_GUIDE.md`](./VPS_MIN_SYNC_GUIDE.md)

## VPS 推奨 `.env`（A+B+C+D 本番）

`deploy/.env.example` を `.env` にコピーし、`GEMINI_API_KEY` を設定してください。

```ini
GEMINI_API_KEY=your_key_here
DINAPOLI_DEFENSE=1
DN_PROP_GATE=1
CHALLENGE_BASE_RISK_PCT_MAX=0.006
DN_PROP_GATE_BASE_RISK_PCT=0.006
DBBS_DEFENSE=1
DBBS_PURE_DATA_MODE=0
DBBS_BEAR_KILL_SWITCH=1
DBBS_BEAR_KILL_SWITCH_THRESHOLD=0.20
DBBS_L2_MIN_SCORE=0
VAMR_DEFENSE=1
VAMR_GEMINI_AUDIT=0
PROFIT_CUSHION_ENABLED=1
PROFIT_CUSHION_TARGET_PCT=2.0
PROFIT_CUSHION_LOT_MULT=0.65
TWIN_BRAKE_ENABLED=1
DD_THROTTLING_ENABLED=1
MUTUAL_EXCLUSION_ENABLED=1
PORTFOLIO_ALLOCATION_ENABLED=0
```

**Profit Cushion ×0.65** は **A/B/C/D 全戦略に共通** の L4.5 防御です。ポートフォリオから D (VAMR) を外した `--strategy abc` でも **同じ 0.65** が適用されます（VAMR 固有の設定ではありません）。

**Portfolio allocation（本番既定 OFF）:** 各戦略は **フル lot**（戦略別ウェイトなし）。3y BT 基準: PF 2.96 / プロップ合格率 100% / 平均合格日数 22.3 日 / WFT stability 90.49。

## Portfolio Strategy Allocation（L4.5 — 任意・戦略別資金配分）

本番は **`PORTFOLIO_ALLOCATION_ENABLED=0`**（ウェイト未適用）。DD をさらに抑えたい場合のみ有効化してください。

| 変数 | 説明 |
|------|------|
| `PORTFOLIO_ALLOCATION_ENABLED=0` | **本番既定** — lot_factor にウェイトを掛けない |
| `PORTFOLIO_ALLOCATION_ENABLED=1` | 最適化ウェイトを lot_factor に反映 |
| `PORTFOLIO_STRATEGY_WEIGHTS_FILE` | JSON ファイル（`deploy/portfolio_allocation_weights.json`） |
| `PORTFOLIO_STRATEGY_WEIGHTS` | インライン JSON（ファイルより優先） |
| `PORTFOLIO_WEIGHT_LSFC` 等 | 個別上書き（setup_type へマップ） |

`portfolio_allocation_optimizer.py` の出力をそのまま反映可能。未登録戦略は **1.0**（従来同等）。ABC のみ運用時は JSON から VAMR 行を削除するか、3 戦略分のみ指定してください。

## L2 — 戦略×シンボル 1 ポジション + ピラミッディング連動

| 層 | 動作 |
|---|---|
| **Python L2** | `one_per_strategy_symbol` — A/B/C/D 各戦略でシンボルごとに最大1ポジション |
| **ピラミッディング** | 有効な戦略は L2 **自動 OFF**（同一戦略内の積み増しを許可。既定: LSFC のみ ON） |
| **MT5 EA** | シンボル単位ブロックなし — Python L2 が正本。`open_positions[]` に `setup_type` / `strategy_letter` を送信 |
| **Bridge JSON** | ポジション comment `PropEA_A` / `PropEA_B` / `PropEA_C` / `PropEA_D` → setup_type へマップ |
| **`.env`** | `MUTUAL_EXCLUSION_ENABLED=1` + `PYRAMID_ENABLED=1` + `PYRAMID_LSFC=1`（`deploy/.env.example`） |

`MUTUAL_EXCLUSION_MODE=daily` / `concurrent` は **廃止**。VPS `.env` から削除してください。

**VPS 反映時:** `PropEA_Bridge.mq5` を **再コンパイル・再アタッチ**（per-strategy L2 + open_positions 拡張）。

## 同梱ファイル（必須）

**DiNapoli (C)**

- `strategies/dinapoli.py` / `dinapoli_mtf.py` / `dinapoli_feature_log.py`
- `src/filters/dn_prop_gate_*.py` / `dn_bayes_ev_v2.py`
- `backtest_results/models/dn_bayes_ev_v2.json`
- `backtest_results/models/dn_prop_gate_v1.json`
- `storage/dn_feature_store.py`

**DBBS (B)**

- `strategies/dbbs.py` / `dbbs_common.py` / `dbbs_bear_kill_switch.py`
- `strategies/scan_numba_util.py`（DBBS 検出 Numba カーネル）
- 本番ペア: **EURUSD, GBPUSD, XAUUSD**

**VAMR (D)**

- `strategies/vamr.py` / `vamr_bayes.py` / `vamr_sizing.py` / `var_reversal.py` / `var_detector.py`
- `backtest_results/models/vamr_bayes_v1.json`
- 本番ペア: **AUDNZD, EURGBP, USDCAD**

**LSFC (A)**

- `strategies/london_sweep_failure.py` / `lsfc_scan_hot.py`

`.gitignore` は `storage/` 配下の実行時 DB を除外しますが、上記 Python モジュールは同期対象です。
