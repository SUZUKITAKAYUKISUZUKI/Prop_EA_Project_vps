"""Export M5 bars from MT5 on VPS to FT6 CSV (run on VPS)."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.market_utils import LIVE_CANONICAL_PAIRS, normalize_pair_name

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed", file=sys.stderr)
    sys.exit(1)

FT6_HEADER = ("<TICKER>", "<DTYYYYMMDD>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>")


def resolve_mt5_symbol(canonical: str) -> str | None:
    for suffix in ("", "p", "P", ".pro", "m", ".a", ".b", ".raw"):
        candidate = f"{canonical}{suffix}" if suffix else canonical
        if mt5.symbol_info(candidate) is not None and mt5.symbol_select(candidate, True):
            return candidate
    for sym in mt5.symbols_get() or []:
        if normalize_pair_name(sym.name) == canonical and mt5.symbol_select(sym.name, True):
            return sym.name
    return None


def export_symbol(canonical: str, output_dir: Path, lookback_days: int) -> dict[str, object]:
    mt5_sym = resolve_mt5_symbol(canonical)
    if not mt5_sym:
        return {"symbol": canonical, "bars": 0, "error": "symbol_not_found"}

    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=lookback_days)
    rates = mt5.copy_rates_range(mt5_sym, mt5.TIMEFRAME_M5, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        return {"symbol": canonical, "bars": 0, "error": f"no_rates:{err}"}

    out = output_dir / f"{canonical}_m5.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(FT6_HEADER)
        for rate in rates:
            dt = datetime.fromtimestamp(int(rate["time"]), tz=timezone.utc)
            writer.writerow(
                [
                    canonical,
                    dt.strftime("%Y%m%d"),
                    dt.strftime("%H%M"),
                    f"{float(rate['open']):.5f}",
                    f"{float(rate['high']):.5f}",
                    f"{float(rate['low']):.5f}",
                    f"{float(rate['close']):.5f}",
                    int(rate["tick_volume"]),
                ]
            )
    return {"symbol": canonical, "bars": len(rates), "file": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export M5 OHLCV from MT5 to FT6 CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbols", default="", help="Comma-separated canonical pairs")
    parser.add_argument("--lookback-days", type=int, default=21)
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"mt5.initialize failed: {mt5.last_error()}", file=sys.stderr)
        return 1

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.symbols.strip():
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = list(LIVE_CANONICAL_PAIRS)

        results = [export_symbol(sym, output_dir, args.lookback_days) for sym in symbols]
        ok = sum(1 for r in results if int(r.get("bars", 0)) > 0)
        print(f"exported {ok}/{len(symbols)} symbols")
        for row in results:
            if row.get("error"):
                print(f"  {row['symbol']}: {row['error']}", file=sys.stderr)
            else:
                print(f"  {row['symbol']}: {row['bars']} bars -> {row.get('file', '')}")
        return 0 if ok > 0 else 1
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
