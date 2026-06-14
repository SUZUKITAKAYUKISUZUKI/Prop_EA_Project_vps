# strategies/archive

本番・バックテストから外したが、参照・再検討用に残すストラテジー実装の保管場所。

## 使い方

- 退役した `.py` をここへ **移動**（ルートの `strategies/` からは削除）
- `backtest_runner` / `strategies/__init__.py` からは **import しない**（`ARCHIVED_STRATEGY_MODES` 登録）
- 必要ならファイル先頭に退役日・理由を 1 行コメントで残す
- 再分析スクリプト・テストは `strategies.archive.*` から import 可

## 格納中

| ファイル | 戦略 | 退役理由 |
|---|---|---|
| **`cspa.py`** ほか `cspa_*` | **CSPA (旧 B)** | **プロップ用ポートフォリオには向いていない**（2026-06-01 アーカイブ） |
| `asian_session_liquidity_sweep.py` | ALS (旧 B) | 取引回数が少ない |
| `fvg_fill.py` | FVG (旧 C) | CSPA 採用に伴いアーカイブ |
| `dtpa.py` | DTPA (F) | 取引回数が少ない |
| `vexp.py` | VEXP (E) | 取引回数が少ない |
| `london_continuation.py` | London Continuation | 本番除外（legacy 順張り） |
| `wyckoff_spring.py` | WS (旧) | 戦略に優位性が無いことが判明したから（後方互換 shim） |
| `wyckoff_reversal.py` | WR (H) | LGR 構築に向けた発展的廃止 |
| `wyckoff_scan_hot.py` | WR scan | WR アーカイブに伴い移動 |
| `wyckoff_scan_numba.py` | WR numba | WR アーカイブに伴い移動 |
| `tokyo_range_expansion_failure.py` | TREF (旧 D) | 単独でのプロップ失格率が高いため |
| `liquidity_grab_reversal.py` | **LGR (I)** | **プロップ向きでない**（自己資金口座向けに優秀） |
| `liquidity_grab_detector.py` | LGR detector | 同上 |
| `lgr_scan_hot.py` / `lgr_scan_numba.py` | LGR scan | 同上 |
| `lgr_detector_debug.py` | LGR debug | 同上 |
| `adre.py` / `adre_detector.py` / `adre_v2_fixed.py` | **ADRE (J)** | **プロップファーム向きではない**（2026-06-13） |
| `reversal_feature_helpers.py` / `_np.py` | LGR/WR 特徴量 | LGR アーカイブに伴い移動 |

LGR 関連サポート（Bayes / EV / Optuna）: `archive/lgr/` — 詳細は `archive/lgr/README.md`

ADRE 関連（Bayes / V2 / validation）: `archive/adre/` — 詳細は `archive/adre/README.md`

CSST 関連（通貨強弱状態遷移 / time exit 研究）: `archive/csst/` — 詳細は `archive/csst/README.md`。**プロップ向きではない**（中長期デイトレ〜スイング向き時間依存型、2026-06-13 アーカイブ）

## 現行 letter マップ（2026-06-01）

| Letter | Mode | 状態 |
|---|---|---|
| **A** | `lsfc` | 本番デフォルト |
| **B** | — | 未割当（旧 CSPA — **アーカイブ**） |
| **C** | `dinapoli` | 本番採用（DN Prop Gate） |
| **D** | — | 未割当（旧 TREF） |
| **H** | `wyckoff` | **アーカイブ** |
| **I** | `lgr` | **アーカイブ** — プロップ向きでない（自己資金口座向けに優秀） |
| **J** | `adre` / `adre_v2` | **アーカイブ** — プロップファーム向きではない |

## 注意

- このフォルダ内のコードは CI・本番 BT の対象外（`ARCHIVED_STRATEGY_MODES` 登録済み）
- 復活させる場合は `strategies/` 直下へ戻し、registry から `ARCHIVED` を外す
- ポートフォリオエイリアス: `abc` / `abcd` / `abcdn` / `ac` = **lsfc + dinapoli**（A+C 共有エクイティ）
