# Prop EA — VPS 最小構成リポジトリ

本フォルダは **開発リポジトリ (`Prop_EA_Project`) から同期される VPS 実運用用の最小セット** です。  
マニフェスト: **`deploy/vps-min-manifest.json` v12**（A+B+C+D+E 本番 + Live Phase 2 + Fintokei 3%）

## 本番ポートフォリオ A+B+C+D+E

| Letter | Mode | 戦略 | Exec / 構造 / ATR | 本番ペア |
|--------|------|------|-------------------|----------|
| **A** | `lsfc` | London Sweep Failure Continuation | M15 / H1 | GBPUSD, EURUSD |
| **B** | `dbbs` | Dual Bollinger Band Squeeze + Bear Kill Switch V2 | M15 / H1 / H4 | **EURUSD, GBPUSD, XAUUSD** |
| **C** | `dinapoli` | DiNapoli Structure + DN Prop Gate V1 | M15 / H1 / H4 | ユニバース |
| **D** | `vamr` | **Volume Area Mean Reversion to POC**（略称 **VAMR**） | H1 / M5 VP / H4 | **AUDNZD, EURGBP, USDCAD** |
| **E** | `smrs` | **Statistical Mean Reversion Scalper**（略称 **SMRS**） | M1 | **AUDNZD, EURGBP, NZDUSD** |

- Python Bridge は `STRATEGY_LETTER_BY_MODE` に登録された **A/B/C/D/E** を同一エクイティで順次評価します。
- **E（SMRS）** — M1 + Phase 3 Bayes + Model A sizing。既定値は `strategies/smrs_production.py` の `PRODUCTION_SPEC`。
- **AUDNZD / EURGBP** は D と E で共有 — **setup_type 単位の L2** で競合を回避。
- BT / WFT / 巨大 CSV / checkpoints は **含みません**（Bayes / DN Prop Gate 用モデル JSON のみ同梱）。

同期手順: [`VPS_MIN_SYNC_GUIDE.md`](./VPS_MIN_SYNC_GUIDE.md)

## Live Phase 2 + Fintokei ルール（v12）

| 機能 | モジュール | 設定 |
|------|-----------|------|
| TP cap 1.5R | `audit/live_tp_cap.py` | `LIVE_TP_CAP_ENABLED=1` |
| DBBS H1 trail（**SL-first / -1R floor**） | `strategies/dbbs_exit.py` + `mt5/DbbsExitManager.mqh` | `DBBS_LIVE_TRAIL_ENABLED=1` |
| Pyramid BE after TP | `live_pyramid/` | `LIVE_PYRAMID_BE_AFTER_TP=1` |
| Fintokei **3% 単一ポジション** lot cap | `audit/risk_manager.py` (`finalize_lot_factor_for_execution`) | `FINTOKEI_SINGLE_POSITION_RULE_ENABLED=1` |

## 3y BT 基準（Live Phase 2 + Fintokei commission, allocation OFF）

| Metric | Value |
|--------|------:|
| Executed trades | 2,409 |
| Total R (effective) | +2,323.52R |
| PF | 2.838 |
| Prop pass rate | **100.00%** |
| Avg pass days | 17.5 |
| Worst window DD | 3.36% |
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
LIVE_TP_CAP_ENABLED=1
LIVE_TP_MAX_R=1.5
DBBS_LIVE_TRAIL_ENABLED=1
LIVE_PYRAMID_BE_AFTER_TP=1
FINTOKEI_SINGLE_POSITION_RULE_ENABLED=1
FINTOKEI_SINGLE_POSITION_LOSS_LIMIT_PCT=3.0
PORTFOLIO_ALLOCATION_ENABLED=0
DBBS_DEFENSE=1
DBBS_BEAR_KILL_SWITCH=1
DINAPOLI_DEFENSE=1
DN_PROP_GATE=1
CHALLENGE_BASE_RISK_PCT_MAX=0.006
VAMR_DEFENSE=1
SMRS_DEFENSE=1
SMRS_GEMINI_AUDIT=0
PYRAMID_SMRS=0
```

## L2 — 戦略×シンボル 1 ポジション + ピラミッディング

| 層 | 動作 |
|---|---|
| **Python L2** | `one_per_strategy_symbol` — A/B/C/D/E 各戦略でシンボルごとに最大1 |
| **ピラミッディング** | 有効戦略は L2 **自動 OFF**（既定: LSFC のみ ON） |
| **MT5 EA** | `open_positions[]` に `setup_type` / `strategy_letter` を送信 |
| **Bridge JSON** | comment `PropEA_A` … `PropEA_E` → setup_type へマップ |

**VPS 反映時:** `PropEA_Bridge.mq5` + `DbbsExitManager.mqh` を **再コンパイル・再アタッチ**（DBBS SL_R_FLOOR / letter E 対応）。

## 同梱ファイル（manifest v12 必須）

**B — DBBS（出口統合）**

- `strategies/dbbs.py` / `dbbs_common.py` / **`dbbs_exit.py`** / `dbbs_bear_kill_switch.py`
- `mt5/DbbsExitManager.mqh` — H1 BB trail + **-1R floor**

**Live 防御（audit/）**

- `live_tp_cap.py` — TP cap / 構造 TP
- `risk_manager.py` — **`finalize_lot_factor_for_execution`**（Fintokei 3% 含む）
- `fintokei_rules.py` — 後方互換 re-export（任意）

**A / C / D / E** — v11 同様（`deploy/VPS_MIN_SYNC_GUIDE.md` 参照）

## VPS 反映チェックリスト

1. 開発 PC: `scripts\sync_vps_min.cmd` → manifest **v12** を確認
2. VPS: `git pull` → `py -3 scripts\vps_bridge_smoke.py` が `[OK]`（phase2 + 3% cap 行も OK）
3. `.env` を `.env.example` どおりに設定
4. MT5: **`PropEA_Bridge.mq5` 再コンパイル**（`DbbsExitManager.mqh` 更新含む）
5. `start_mt5_bridge.bat` 再起動
