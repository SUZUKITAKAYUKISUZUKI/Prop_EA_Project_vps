# VPS 最小構成 — 初回セットアップ & 同期手順

開発環境 (`Prop_EA_Project`) から、VPS 実運用用の最小フォルダ (`Prop_EA_Project_vps`) へ **同名ファイルだけ** をコピー同期します。

| 役割 | フォルダ | Git |
|------|----------|-----|
| 開発 | `C:\Prop_EA_Project` | 全量（BT 結果は .gitignore 推奨） |
| VPS 最小 | `C:\Prop_EA_Project_vps` | **GitHub 用リポジトリ**（軽量） |

---

## 事前準備（1 回だけ）

### A. 同期スクリプトの場所を確認

1. エクスプローラーを開く
2. アドレスバーに `C:\Prop_EA_Project\scripts` と入力 → Enter
3. 次の 2 ファイルがあることを確認する  
   - `sync_vps_min.ps1`  
   - `sync_vps_min.cmd` ← **ダブルクリック用**

### B. PowerShell 実行ポリシー（初回のみ・管理者不要）

1. **Windows キー** を押す  
2. `PowerShell` と入力  
3. **Windows PowerShell** を右クリック → **管理者として実行**  
4. 次を貼り付け → Enter  

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

5. `Y` → Enter  
6. ウィンドウを閉じる  

> `.cmd` から `-ExecutionPolicy Bypass` で呼ぶため、通常は A 不要です。

---

## 初回：最小フォルダ作成 & GitHub リポジトリ

### ステップ 1 — 最小構成をコピー

1. エクスプローラーで `C:\Prop_EA_Project\scripts` を開く  
2. **`sync_vps_min.cmd` をダブルクリック**  
3. 黒い窓に `[copied]` が並ぶのを待つ  
4. `=== Sync complete ===` と `Target : C:\Prop_EA_Project_vps` を確認  
5. 何かキーを押して窓を閉じる  

**確認:** エクスプローラーで `C:\Prop_EA_Project_vps` ができ、`main_platform.py` があること。

---

### ステップ 2 — GitHub に空リポジトリを作る

1. ブラウザで [https://github.com/new](https://github.com/new) を開く  
2. **Repository name** に例: `Prop_EA_Project_vps` と入力  
3. **Public / Private** を選ぶ  
4. **Add a README** は **オフ**（空リポジトリ）  
5. **Create repository** をクリック  
6. 表示された URL を控える（例: `https://github.com/SUZUKITAKAYUKISUZUKI/Prop_EA_Project_vps.git`）

---

### ステップ 3 — 最小フォルダを Git 初期化

1. **Windows キー** → `PowerShell` → Enter  
2. 次を **1 行ずつ** 実行  

```powershell
cd C:\Prop_EA_Project_vps
git init
git add .
git commit -m "Initial VPS minimum deploy"
git branch -M main
git remote add origin https://github.com/SUZUKITAKAYUKISUZUKI/Prop_EA_Project_vps.git
git push -u origin main
```

3. GitHub のユーザー名・パスワード（または PAT）を求められたら入力  

**確認:** GitHub 上のリポジトリに `.py` / `mt5/` 等が見えること（数百 KB〜数 MB 程度）。

---

## 日常：開発環境を変更したあと同期 → GitHub

コードを直した **たびに** 次の流れです。

### ステップ 1 — 開発側で保存

1. Cursor / VS Code でファイルを編集  
2. **Ctrl + S** で保存  

---

### ステップ 2 — 最小構成へ同期（1 クリック）

1. エクスプローラー → `C:\Prop_EA_Project\scripts`  
2. **`sync_vps_min.cmd` をダブルクリック**  
3. 変更があったファイルだけ `[copied]` と表示される（同一内容はスキップ）  
4. キー押下で閉じる  

**オプション（PowerShell）:**

```powershell
cd C:\Prop_EA_Project
.\scripts\sync_vps_min.ps1 -Clean
```

`-Clean` … 最小構成に **含まれないファイル** を VPS フォルダから削除（整理用）。

---

### ステップ 3 — 最小構成を GitHub に push

1. PowerShell を開く  
2. 実行  

```powershell
cd C:\Prop_EA_Project_vps
git status
git add -u
git add .
git commit -m "Sync from dev: 変更内容の短い説明"
git push origin main
```

**確認:** GitHub リポジトリの最新コミット日時が更新されていること。

---

## VPS サーバーへ反映（デプロイ）

1. VPS に RDP 接続  
2. PowerShell  

```powershell
cd C:\Prop_EA_Project_vps
git pull origin main
copy .env.example .env
notepad .env
```

3. `.env` に `GEMINI_API_KEY` を記入 → 保存  
4. `.env` を **A+B+C 本番** に合わせる（`deploy/.env.example` 参照）  
   - `DINAPOLI_DEFENSE=1` / `DN_PROP_GATE=1`  
   - `DBBS_DEFENSE=1` / `DBBS_BEAR_KILL_SWITCH=1`  
   - **`MUTUAL_EXCLUSION_MODE=daily` があれば削除**（`MUTUAL_EXCLUSION_ENABLED=1` のみ）  
5. `start_mt5_bridge.bat` をダブルクリック（再起動）  
6. MT5 で **`PropEA_Bridge.mq5` を再コンパイル・再アタッチ**（`open_positions` 送信対応）

### L2 — 戦略×シンボル 1 ポジション（A+B+C）

| 層 | 役割 |
|---|---|
| Python | `one_per_strategy_symbol` — A/B/C 各戦略で EURUSD・GBPUSD に最大1 |
| ピラミッディング | 有効戦略は L2 自動 OFF（既定: `PYRAMID_LSFC=1` のみ） |
| MT5 EA | シンボル単位ブロックなし — `open_positions[]` に setup_type / letter |
| `.env` | `MUTUAL_EXCLUSION_ENABLED=1` + `PYRAMID_ENABLED=1`（`deploy/.env.example`） |

**必須:** MT5 で `PropEA_Bridge.mq5` を再コンパイル（per-strategy L2 対応版）。

---

## 同期対象のカスタマize

一覧は **`C:\Prop_EA_Project\deploy\vps-min-manifest.json`** で管理します。

| 操作 | 手順 |
|------|------|
| ファイルを増やす | `root_files` にパス追加 → 保存 → `sync_vps_min.cmd` |
| フォルダごと増やす | `directories_all_files` に追加 |
| 除外する | `exclude_globs` に追加 |

---

## トラブルシュート

| 症状 | 対処 |
|------|------|
| スクリプトが実行できない | ステップ「事前準備 B」または `.cmd` を使用 |
| `Target` が別ドrive になりたい | `sync_vps_min.cmd` を右クリック → ショートカット作成 → リンク先に `-TargetDir D:\Prop_EA_Project_vps` を追加 |
| push が大きすぎる | 開発側の `backtest_results/` / `data/` が VPS 側に入っていないか確認。`-Clean` 実行 |
| VPS で import エラー | 同期後 `pip install -r requirements.txt` |
| MT5 → `/trade_signal` が **500** | 下記「500 エラー」を参照 |

---

## VPS: `/trade_signal` が 500 のとき

uvicorn のアクセスログだけでは原因が分かりません。**ブリッジ窓**に `POST /trade_signal failed` と traceback が出ます（v1.1 以降）。

### 手順 1 — スモークテスト（VPS 上）

```powershell
cd C:\Prop_EA_Project_vps
py -3 scripts\vps_bridge_smoke.py
```

| 結果 | 意味 |
|------|------|
| `[FAIL] strategies/bt_ohlcv.py is missing` 等 | 最小同期が古い → 下記「手順 2」 |
| `[FAIL] evaluate_trade_signal raised` | 表示された traceback が本番エラーと同じ |
| `[OK] evaluate_trade_signal` | Python 側は正常 → MT5 の JSON / 二重 EA を疑う |

### 手順 2 — よくある原因: 最小構成の未同期

A+B+C 本番（LSFC + DBBS + DiNapoli）では次が必須です。

**A — LSFC**

- `strategies/london_sweep_failure.py` / `lsfc_scan_hot.py`

**B — DBBS**

- `strategies/dbbs.py` / `dbbs_common.py` / `dbbs_bear_kill_switch.py`
- `strategies/scan_numba_util.py`

**C — DiNapoli**

- `strategies/dinapoli.py` / `dinapoli_mtf.py` / `dinapoli_feature_log.py`
- `src/filters/dn_prop_gate_*.py` / `dn_bayes_ev_v2.py`
- `storage/dn_feature_store.py`
- `backtest_results/models/dn_*.json`

旧マニフェスト（v1/v2）は `src/` や DBBS / DN モデルを含まず、import エラー → **500** になります。

1. 開発 PC で `sync_vps_min.cmd` を実行（`deploy/vps-min-manifest.json` **v4**）
2. VPS で `git pull` または手動で上記ファイルを配置
3. `.env` を `deploy/.env.example` どおり A+B+C に設定
4. `start_mt5_bridge.bat` を再起動

### 手順 3 — 500 の本文を確認（任意）

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/trade_signal -ContentType "application/json" -Body '{"market":{"pair":"GBPUSD","open":1.27,"high":1.271,"low":1.269,"close":1.27,"volume":100},"calendar":{"minutes_to_next_news":45,"news_impact_level":"HIGH"},"account":{"equity":100000,"balance":100000},"bar_time":"2026-06-01 12:00:00","server_time":"2026-06-01 12:00:00","spread_points":10}'
```

`detail` フィールドに Python 例外メッセージが入ります。

---

## クイック参照（コピペ用）

```powershell
# ① 開発 → 最小 同期
cd C:\Prop_EA_Project
.\scripts\sync_vps_min.ps1

# ② 最小 → GitHub
cd C:\Prop_EA_Project_vps
git add -u; git add .
git commit -m "Sync from dev"
git push origin main

# ③ VPS で取得
cd C:\Prop_EA_Project_vps
git pull origin main
```
