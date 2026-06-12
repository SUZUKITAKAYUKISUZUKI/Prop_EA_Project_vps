# Prop EA — VPS 最小構成リポジトリ

本フォルダは **開発リポジトリ (`Prop_EA_Project`) から同期される VPS 実運用用の最小セット** です。

- 本番戦略: **LSFC (A) + Dual Bollinger Band Squeeze (B) + DiNapoli (C)** — 共有エクイティ A+B+C ポートフォリオ
- **CSPA (旧 B)** はアーカイブ（VPS Live 対象外。`main_platform` 互換のため archive コードのみ同梱）
- BT / WFT / 巨大 CSV / checkpoints は **含みません**（DN Prop Gate 用モデル JSON のみ同梱）

同期手順の詳細: [`VPS_MIN_SYNC_GUIDE.md`](./VPS_MIN_SYNC_GUIDE.md)

## VPS 推奨 `.env`（A+B+C 本番）

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
PROFIT_CUSHION_ENABLED=1
TWIN_BRAKE_ENABLED=1
DD_THROTTLING_ENABLED=1
MUTUAL_EXCLUSION_ENABLED=1
```

## L2 — 同一シンボル1ポジション（VPS 本番）

| 層 | 動作 |
|---|---|
| **Python L2** | `open_positions` 同期 + `MUTUAL_EXCLUSION_LOCK`（LSFC / DiNapoli 不問） |
| **MT5 EA** | `HasOpenPosition(symbol)` — 二重安全弁 |
| **Bridge JSON** | 毎リクエスト `open_positions[]` を送信（`PropEA_Bridge.mq5`） |

`MUTUAL_EXCLUSION_MODE=daily` / `concurrent` は **廃止**。VPS `.env` から削除してください。

## 同梱ファイル（DiNapoli Prop Gate）

- `backtest_results/models/dn_bayes_ev_v2.json`
- `backtest_results/models/dn_prop_gate_v1.json`
- `storage/dn_feature_store.py`（特徴量カラム定義のみ。`.db` は VPS 上で自動生成可）

`.gitignore` は `storage/` 配下の実行時 DB を除外しますが、上記 Python モジュールは同期対象です。