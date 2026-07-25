"""Fetch declared corporate actions for universe names and rebuild the adjusted price panel.

Exists because detecting splits from the exchange's restated previous close proved unreliable:
NSE leaves PREV_CLOSE un-restated for a large share of genuine capital changes, so detection alone
left fictitious one-day crashes of 50-90% in the return series. Declared split ratios do not
depend on the exchange having restated anything.

Every declared factor is still reconciled against the price move actually observed on its ex-date.
Events the prices corroborate are applied; events they contradict are written out for review and
deliberately NOT applied, because applying an unverified factor injects error rather than removing
it. Both counts are reported.

Usage:
    python scripts/build_corporate_actions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.io import write_derived_parquet  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.data.corporate_actions import (  # noqa: E402
    apply_adjustments,
    fetch_declared_splits,
    find_unexplained_moves,
    reconcile_with_prices,
    resolve_disputed,
)

_log = get_logger("build_corporate_actions")


def main() -> int:
    cfg = load_config()
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")

    # Only universe names matter: they are what any strategy can actually trade, and fetching
    # every listed symbol would be a long network run for data nothing downstream reads.
    symbols = sorted(universe["symbol"].unique().to_list())
    _log.info("fetching declared corporate actions for %d universe symbols", len(symbols))

    with RunManifest(cfg, script="scripts/build_corporate_actions.py") as run:
        declared = []
        failed: list[str] = []
        for index, symbol in enumerate(symbols, start=1):
            try:
                declared.extend(
                    fetch_declared_splits(symbol, cfg.calendar.history_start, cfg.dates.dev_end)
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not abort the run
                failed.append(f"{symbol}: {type(exc).__name__}")
            if index % 50 == 0:
                _log.info("%d/%d symbols fetched", index, len(symbols))

        agreed, disputed = reconcile_with_prices(declared, panel, calendar)

        # Resolve contradicted events by snapping the observed ratio to a conventional factor.
        # Only dates the feed already flagged are eligible — see resolve_disputed for why.
        snapped, decisions = resolve_disputed(disputed)

        base = panel.drop([c for c in panel.columns if c.startswith("adj_") or c == "divisor"])
        adjusted = apply_adjustments(base, agreed + snapped)
        write_derived_parquet(adjusted, cfg.paths.data_processed / "prices_adjusted.parquet")

        # The snap table is the reviewer's verification surface: every judgement in one place.
        if decisions:
            write_derived_parquet(
                pl.DataFrame({
                    "symbol": [d.symbol for d in decisions],
                    "ex_date": [d.ex_date for d in decisions],
                    "declared_factor": [d.declared_factor for d in decisions],
                    "implied_factor": [d.implied_factor for d in decisions],
                    "applied_factor": [d.applied_factor for d in decisions],
                    "outcome": [d.outcome for d in decisions],
                }),
                cfg.paths.data_processed / "snap_table.parquet",
            )

        # Anything large the feed never mentioned stays untouched and is surfaced, not corrected.
        declared_dates = {(e.symbol, e.ex_date) for e in declared}
        unexplained = find_unexplained_moves(adjusted, calendar, declared_dates)
        write_derived_parquet(
            unexplained, cfg.paths.data_processed / "unexplained_moves.parquet"
        )

        run.note("symbols_queried", len(symbols))
        run.note("declared_events", len(declared))
        run.note("applied_events", len(agreed))
        run.note("disputed_events", len(disputed))
        run.note("snapped_events", len(snapped))
        run.note("unresolved_events", sum(1 for d in decisions if d.outcome == "unresolved"))
        run.note("unexplained_moves", unexplained.height)
        run.note("symbol_fetch_failures", failed)

    _log.info("declared=%d applied=%d snapped=%d unresolved=%d unexplained_moves=%d",
              len(declared), len(agreed), len(snapped),
              sum(1 for d in decisions if d.outcome == "unresolved"), unexplained.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
