# VPS 最小構成 — 初回セットアップ & 同期手順

開発環境 (`Prop_EA_Project`) から、VPS 実運用用の最小フォルダ (`Prop_EA_Project_vps`) へ **同名ファイルだけ** をコピー同期します。

| 役割 | フォルダ | Git |
|------|----------|-----|
| 開発 | `C:\Prop_EA_Project` | 全量（BT 結果は .gitignore 推奨） |
| VPS 最小 | `C:\Prop_EA_Project_vps` | **GitHub 用リポジトリ**（軽量） |

**本番ポートフォリオ:** A+B+C+D+E（LSFC + DBBS + DiNapoli + VAMR + SMRS Model A）  
**マニフェスト:** `deploy/vps-min-manifest.json` **v12**  
**Live Phase 2:** TP cap 1.5R / DBBS H1 trail（SL-first, -1R floor）/ pyramid BE / Fintokei 3% single-position lot cap

---

## 事前準備（1 回だけ）

### A. 同期スクリプトの場所を確認

1. エクスプローラーで `C:\Prop_EA_Project\scripts` を開く
2. 次の 2 ファイルがあることを確認する
   - `sync_vps_min.ps1`
   - `sync_vps_min.cmd` ← **ダブルクリック用**

### B. PowerShell 実行ポリシー（初回のみ・管理者不要）

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

> `.cmd` から `-ExecutionPolicy Bypass` で呼ぶため、通常は不要です。

---

## 初回：最小フォルダ作成 & GitHub リポジトリ

### ステップ 1 — 最小構成をコピー

1. `C:\Prop_EA_Project\scripts\sync_vps_min.cmd` を **ダブルクリック**
2. 先頭に `Manifest: deploy/vps-min-manifest.json v12` と表示されることを確認
3. `[copied]` が並び、`=== Sync complete ===` まで待つ

**確認:** `C:\Prop_EA_Project_vps` に以下があること

- `main_platform.py`
- `strategies/dbbs_exit.py`
- `audit/live_tp_cap.py`
- `mt5/DbbsExitManager.mqh`
- `backtest_results/models/smrs_bayes_v1.json`
- `deploy/portfolio_allocation_weights.json`

---

### ステップ 2 — GitHub に空リポジトリを作る

1. [https://github.com/new](https://github.com/new) で空リポジトリを作成
2. URL を控える（例: `https://github.com/USER/Prop_EA_Project_vps.git`）

---

### ステップ 3 — 最小フォルダを Git 初期化

```powershell
cd C:\Prop_EA_Project_vps
git init
git add .
git commit -m "Initial VPS minimum deploy (manifest v12, ABCDE Live Phase2)"
git branch -M main
git remote add origin https://github.com/USER/Prop_EA_Project_vps.git
git push -u origin main
```

---

## 日常：開発環境を変更したあと同期 → GitHub

### ステップ 1 — 開発側で保存（Ctrl + S）

### ステップ 2 — 最小構成へ同期

`sync_vps_min.cmd` をダブルクリック。

**PowerShell オプション:**

```powershell
cd C:\Prop_EA_Project
.\scripts\sync_vps_min.ps1 -Clean
```

`-Clean` … マニフェスト外ファイルを VPS フォルダから削除（整理用）。

### ステップ 3 — GitHub に push

```powershell
cd C:\Prop_EA_Project_vps
git add -u
git add .
git commit -m "Sync from dev: 変更内容の短い説明"
git push origin main
```

---

## VPS サーバーへ反映（デプロイ）

```powershell
cd C:\Prop_EA_Project_vps
git pull origin main
copy .env.example .env
notepad .env
```

### `.env` 必須設定（A+B+C+D+E + Live Phase 2）

`deploy/.env.example` を参照。最低限:

| カテゴリ | 変数 |
|----------|------|
| API | `GEMINI_API_KEY` |
| 共通防御 | `PROFIT_CUSHION_ENABLED=1` / `PROFIT_CUSHION_LOT_MULT=0.65` / `TWIN_BRAKE_ENABLED=1` / `DD_THROTTLING_ENABLED=1` |
| L2 | `MUTUAL_EXCLUSION_ENABLED=1` / `PYRAMID_ENABLED=1` / `PYRAMID_LSFC=1` |
| Live Phase 2 | `LIVE_TP_CAP_ENABLED=1` / `LIVE_TP_MAX_R=1.5` / `DBBS_LIVE_TRAIL_ENABLED=1` / `LIVE_PYRAMID_BE_AFTER_TP=1` |
| Fintokei 3% | **`FINTOKEI_SINGLE_POSITION_RULE_ENABLED=1`** / `FINTOKEI_SINGLE_POSITION_LOSS_LIMIT_PCT=3.0` |
| 配分 | **`PORTFOLIO_ALLOCATION_ENABLED=0`**（本番既定） |
| B DBBS | `DBBS_DEFENSE=1` / `DBBS_BEAR_KILL_SWITCH=1` |
| C DiNapoli | `DINAPOLI_DEFENSE=1` / `DN_PROP_GATE=1` / `CHALLENGE_BASE_RISK_PCT_MAX=0.006` |
| D VAMR | `VAMR_DEFENSE=1` / `VAMR_GEMINI_AUDIT=0` |
| E SMRS | `SMRS_DEFENSE=1` / `SMRS_GEMINI_AUDIT=0` / `PYRAMID_SMRS=0` |

> **`MUTUAL_EXCLUSION_MODE=daily` があれば削除**（廃止済み）。

### デプロイ後チェック

```powershell
cd C:\Prop_EA_Project_vps
py -3 scripts\vps_bridge_smoke.py
```

| 結果 | 意味 |
|------|------|
| `[OK] VPS minimum files present (A+B+C+D+E)` | manifest v12 同期 OK |
| `[OK] Strategy registry (letter E / abcde expansion)` | Python レジストリ OK |
| `[OK] Live phase2 exits + Fintokei 3% lot cap` | DBBS exit / lot_factor 統合 OK |
| `[OK] evaluate_trade_signal` | Bridge 評価パス OK |
| `[FAIL] Missing ...` | 開発 PC で `sync_vps_min.cmd` を再実行 → push → pull |

### MT5 / Bridge 再起動

1. MT5 で **`PropEA_Bridge.mq5` を再コンパイル・再アタッチ**（`DbbsExitManager.mqh` v12 = SL_R_FLOOR）
2. `start_mt5_bridge.bat` をダブルクリック（再起動）

---

## manifest v12 追加・更新モジュール

| モジュール | 用途 |
|-----------|------|
| `strategies/dbbs_exit.py` | DBBS H1 trail / Live JSON / BT L5 出口 |
| `audit/live_tp_cap.py` | Live TP cap 1.5R + 構造 TP |
| `audit/risk_manager.py` | `finalize_lot_factor_for_execution`（Fintokei 3%） |
| `mt5/DbbsExitManager.mqh` | Live DBBS trail + **-1R floor** |

### モデル JSON（`backtest_results/models/`）

| ファイル | 用途 |
|----------|------|
| `dn_bayes_ev_v2.json` / `dn_prop_gate_v1.json` | Strategy C |
| `vamr_bayes_v1.json` | Strategy D |
| `smrs_bayes_v1.json` | Strategy E |

### 意図的に除外（BT のみ — VPS 不要）

- `strategies/smrs_portfolio.py`
- `strategies/bt_l5*.py`, `bt_scan_parallel.py`
- `audit/live_exit_bt.py`（BT 専用 — Live 不要）

---

## 同期対象のカスタマイズ

`deploy/vps-min-manifest.json` を編集 → `sync_vps_min.cmd` 再実行。

| 操作 | 手順 |
|------|------|
| ファイルを増やす | `root_files` にパス追加 |
| フォルダごと増やす | `directories_all_files` に追加 |
| 除外する | `exclude_globs` に追加 |

---

## トラブルシュート

| 症状 | 対処 |
|------|------|
| スクリプトが実行できない | `.cmd` を使用、または実行ポリシー B |
| push が大きすぎる | `backtest_results/` CSV が混入していないか確認 → `-Clean` |
| VPS で import エラー | `pip install -r requirements.txt` → smoke test |
| MT5 → `/trade_signal` **500** | 下記参照 |
| DBBS が旧 Fixed SL/TP のまま | `dbbs_exit.py` + `DbbsExitManager.mqh` 同期 → **再コンパイル** |
| manifest v11 未満 | `dbbs_exit.py` / Fintokei 3% cap 欠落 → v12 同期 |

---

## VPS: `/trade_signal` が 500 のとき

### 手順 1 — スモークテスト

```powershell
cd C:\Prop_EA_Project_vps
py -3 scripts\vps_bridge_smoke.py
```

### 手順 2 — よくある原因: 古い最小構成

manifest **v12 未満** では `dbbs_exit.py` / `live_tp_cap.py` / 更新 `risk_manager.py` が欠け、import エラー → **500** になります。

1. 開発 PC で `sync_vps_min.cmd`（**v12** 表示を確認）
2. VPS で `git pull`
3. `.env` を Live Phase 2 + Fintokei 3% 含む本番設定に合わせる
4. smoke test → MT5 再コンパイル → bridge 再起動

---

## クイック参照（コピペ用）

```powershell
# ① 開発 → 最小 同期
cd C:\Prop_EA_Project
.\scripts\sync_vps_min.ps1

# ② 最小 → GitHub
cd C:\Prop_EA_Project_vps
git add -u; git add .
git commit -m "Sync from dev (manifest v12 ABCDE Live Phase2)"
git push origin main

# ③ VPS で取得 & 検証
cd C:\Prop_EA_Project_vps
git pull origin main
py -3 scripts\vps_bridge_smoke.py
```
