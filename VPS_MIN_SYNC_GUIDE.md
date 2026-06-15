# VPS 最小構成 — 初回セットアップ & 同期手順

開発環境 (`Prop_EA_Project`) から、VPS 実運用用の最小フォルダ (`Prop_EA_Project_vps`) へ **同名ファイルだけ** をコピー同期します。

| 役割 | フォルダ | Git |
|------|----------|-----|
| 開発 | `C:\Prop_EA_Project` | 全量（BT 結果は .gitignore 推奨） |
| VPS 最小 | `C:\Prop_EA_Project_vps` | **GitHub 用リポジトリ**（軽量） |

**本番ポートフォリオ:** A+B+C+D+E（LSFC + DBBS + DiNapoli + VAMR + Statistical Mean Reversion Scalper / SMRS Model A）  
**マニフェスト:** `deploy/vps-min-manifest.json` **v8**

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
2. 先頭に `Manifest: deploy/vps-min-manifest.json v8` と表示されることを確認
3. `[copied]` が並び、`=== Sync complete ===` まで待つ

**確認:** `C:\Prop_EA_Project_vps` に以下があること

- `main_platform.py`
- `backtest_results/models/smrs_bayes_v1.json`
- `backtest_results/models/vamr_bayes_v1.json`
- `strategies/smrs_production.py`
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
git commit -m "Initial VPS minimum deploy (manifest v8, ABCDE)"
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

### `.env` 必須設定（A+B+C+D+E）

`deploy/.env.example` を参照。最低限:

| カテゴリ | 変数 |
|----------|------|
| API | `GEMINI_API_KEY` |
| 共通防御 | `PROFIT_CUSHION_ENABLED=1` / `PROFIT_CUSHION_LOT_MULT=0.65` / `TWIN_BRAKE_ENABLED=1` / `DD_THROTTLING_ENABLED=1` |
| L2 | `MUTUAL_EXCLUSION_ENABLED=1` / `PYRAMID_ENABLED=1` / `PYRAMID_LSFC=1` |
| 配分 | **`PORTFOLIO_ALLOCATION_ENABLED=0`**（本番既定） |
| B DBBS | `DBBS_DEFENSE=1` / `DBBS_BEAR_KILL_SWITCH=1` |
| C DiNapoli | `DINAPOLI_DEFENSE=1` / `DN_PROP_GATE=1` |
| D Volume Area Mean Reversion to POC (VAMR) | `VAMR_DEFENSE=1` / `VAMR_GEMINI_AUDIT=0` |
| E Statistical Mean Reversion Scalper (SMRS) | `SMRS_GEMINI_AUDIT=0` / `SMRS_LLM_AUDIT=0` |

> **`MUTUAL_EXCLUSION_MODE=daily` があれば削除**（廃止済み）。

### デプロイ後チェック

```powershell
cd C:\Prop_EA_Project_vps
py -3 scripts\vps_bridge_smoke.py
```

| 結果 | 意味 |
|------|------|
| `[OK] VPS minimum files present (A+B+C+D+E)` | manifest v8 同期 OK |
| `[OK] Strategy registry (letter E / abcde expansion)` | Python レジストリ OK |
| `[OK] evaluate_trade_signal` | Bridge 評価パス OK |
| `[FAIL] Missing ...` | 開発 PC で `sync_vps_min.cmd` を再実行 → push → pull |

### MT5 / Bridge 再起動

1. MT5 で **`PropEA_Bridge.mq5` を再コンパイル・再アタッチ**（letter **E** = SMRS 対応）
2. `start_mt5_bridge.bat` をダブルクリック（再起動）

---

## L2 — 戦略×シンボル 1 ポジション（A+B+C+D+E）

| 層 | 役割 |
|---|---|
| Python | `one_per_strategy_symbol` — 各戦略×シンボル最大1 |
| ピラミッディング | 有効戦略は L2 自動 OFF（既定: LSFC のみ ON） |
| MT5 EA | `open_positions[]` に `setup_type` / `strategy_letter` (A–E) |
| 共有ペア | AUDNZD / EURGBP は **D (VAMR)** と **E (SMRS)** で別 setup_type — L2 は戦略単位 |

---

## manifest v8 同梱一覧

### モデル JSON（`backtest_results/models/`）

| ファイル | 用途 |
|----------|------|
| `dn_bayes_ev_v2.json` / `dn_prop_gate_v1.json` | Strategy C |
| `vamr_bayes_v1.json` | Strategy D — Volume Area Mean Reversion to POC (VAMR) |
| `smrs_bayes_v1.json` | Strategy E — Statistical Mean Reversion Scalper (SMRS) |

### 戦略 Python（抜粋）

| Letter | 必須モジュール |
|--------|----------------|
| A | `london_sweep_failure.py`, `lsfc_scan_hot.py` |
| B | `dbbs.py`, `dbbs_common.py`, `dbbs_bear_kill_switch.py`, `scan_numba_util.py` |
| C | `dinapoli.py`, `dinapoli_mtf.py`, `dinapoli_universe_fast.py`, `src/filters/dn_prop_gate_*` |
| D | `vamr.py`, `vamr_bayes.py`, `vamr_features.py`, `vamr_phase2.py`, `var_reversal.py`, `var_detector.py` |
| E | `smrs_pure.py`, `smrs_scan_numba.py`, `smrs_bayes.py`, `smrs_production.py` |

### 意図的に除外（BT のみ — VPS 不要）

- `strategies/smrs_portfolio.py`
- `strategies/smrs_sizing.py`
- `strategies/bt_l5*.py`, `bt_scan_parallel.py`

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
| letter E が MT5 comment に出ない | `PropEA_Bridge.mq5` を v8 版で再コンパイル |

---

## VPS: `/trade_signal` が 500 のとき

### 手順 1 — スモークテスト

```powershell
cd C:\Prop_EA_Project_vps
py -3 scripts\vps_bridge_smoke.py
```

### 手順 2 — よくある原因: 古い最小構成

manifest **v8 未満** では VAMR / SMRS モデルや `vamr_phase2.py` が欠け、import エラー → **500** になります。

1. 開発 PC で `sync_vps_min.cmd`（v8 表示を確認）
2. VPS で `git pull`
3. `.env` を A+B+C+D+E 本番に合わせる
4. smoke test → bridge 再起動

### 手順 3 — 500 本文を確認（任意）

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/trade_signal -ContentType "application/json" -Body '{"market":{"pair":"GBPUSD","open":1.27,"high":1.271,"low":1.269,"close":1.27,"volume":100},"calendar":{"minutes_to_next_news":45,"news_impact_level":"HIGH"},"account":{"equity":100000,"balance":100000},"bar_time":"2026-06-01 12:00:00","server_time":"2026-06-01 12:00:00","spread_points":10}'
```

---

## クイック参照（コピペ用）

```powershell
# ① 開発 → 最小 同期
cd C:\Prop_EA_Project
.\scripts\sync_vps_min.ps1

# ② 最小 → GitHub
cd C:\Prop_EA_Project_vps
git add -u; git add .
git commit -m "Sync from dev (manifest v8 ABCDE)"
git push origin main

# ③ VPS で取得 & 検証
cd C:\Prop_EA_Project_vps
git pull origin main
py -3 scripts\vps_bridge_smoke.py
```
