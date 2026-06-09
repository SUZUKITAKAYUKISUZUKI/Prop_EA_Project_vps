# strategies/archive

本番・バックテストから外したが、参照・再検討用に残すストラテジー実装の保管場所。

## 使い方

- 退役した `.py` をここへ **移動**（ルートの `strategies/` からは削除）
- `backtest_runner` / `strategies/__init__.py` からは **import しない**
- 必要ならファイル先頭に退役日・理由を 1 行コメントで残す
- 再分析スクリプト・テストは `strategies.archive.wyckoff_reversal` 等から import 可

## 格納中

| ファイル | 戦略 | 退役理由 |
|---|---|---|
| `asian_session_liquidity_sweep.py` | ALS (旧 B) | 取引回数が少ない |
| `fvg_fill.py` | FVG (旧 C) | CSPA (B) 正式採用に伴いアーカイブ |
| `dtpa.py` | DTPA (F) | 取引回数が少ない |
| `vexp.py` | VEXP (E) | 取引回数が少ない |
| `london_continuation.py` | London Continuation | 本番除外（legacy 順張り） |
| `wyckoff_spring.py` | WS (旧) | 戦略に優位性が無いことが判明したから（後方互換 shim） |
| `wyckoff_reversal.py` | WR (H) | 新戦略 Liquidity Grab Reversal (LGR) 構築に向けての発展的廃止 |
| `wyckoff_scan_hot.py` | WR scan | WR アーカイブ（LGR 構築に向けた発展的廃止）に伴い移動 |
| `wyckoff_scan_numba.py` | WR numba | WR アーカイブ（LGR 構築に向けた発展的廃止）に伴い移動 |
| `tokyo_range_expansion_failure.py` | TREF (旧 D) | 単独でのプロップ失格率が高いため |

## 現行 letter マップ（2026-06-01）

| Letter | Mode | 状態 |
|---|---|---|
| **A** | `lsfc` | 本番デフォルト |
| **B** | `cspa` | 本番採用（WFT 検証済み） |
| **C** | — | 未割当（旧 FVG） |
| **D** | — | 未割当（旧 TREF） |
| **H** | `wyckoff` | **アーカイブ** — LGR 構築に向けた発展的廃止（旧 WR） |
| **I** | `lgr` | Liquidity Grab Reversal (LGR) — BT/WFT のみ |

## 注意

- このフォルダ内のコードは CI・本番 BT の対象外（`ARCHIVED_STRATEGY_MODES` に `wyckoff` 登録済み）
- 復活させる場合は `strategies/` 直下へ戻し、registry へ再登録する
- ポートフォリオエイリアス: `abc` / `abcd` = **lsfc + cspa**（A+B 共有エクイティ）
- LGR 特徴量ヘルパーは `strategies/reversal_feature_helpers.py`（WR 非依存）
