"""Measure how far the authoritative price panel diverges from an independent source.

Compares raw (unadjusted) closes on purpose. Comparing adjusted series would conflate two
different questions — whether the underlying prices agree, and whether two adjustment
methodologies agree — and only the first tells us whether the panel can be trusted at all.

Usage:
    python scripts/cross_check_prices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.data.prices import cross_check  # noqa: E402

_log = get_logger("cross_check_prices")

SAMPLE_SIZE = 30


def main() -> int:
    cfg = load_config()
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")

    # The most persistent universe members: the core liquid names a strategy would actually hold.
    counts = universe.group_by("symbol").len().sort("len", descending=True)
    symbols = counts.head(SAMPLE_SIZE)["symbol"].to_list()

    window = panel.filter(
        (pl.col("session_date") >= cfg.dates.dev_start)
        & (pl.col("session_date") <= cfg.dates.dev_end)
    )
    result = cross_check(window, symbols, cfg.data.prices.cross_check_rel_tol)

    print(f"symbols compared:  {result.symbols_compared} of {len(symbols)}")
    print(f"points compared:   {result.compared_points:,}")
    print(f"mismatches:        {result.mismatches:,}")
    print(f"discrepancy rate:  {result.discrepancy_rate:.4%}")
    print("\nworst offenders (symbol, date, bhavcopy, reference, rel diff):")
    for symbol, day, mine, theirs, diff in result.worst:
        print(f"  {symbol:12s} {day}  {mine:10.2f}  {theirs:10.2f}  {diff:7.2%}")

    # A cluster of mismatches concentrated in a few names is the signature of a corporate action
    # handled differently by the two sources, not of noisy prices. Report that shape explicitly.
    by_symbol: dict[str, int] = {}
    for symbol, *_ in result.worst:
        by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
    print(f"\nsymbols appearing among the worst offenders: {sorted(by_symbol)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
