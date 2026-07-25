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
    reconcile_with_prices,
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
        adjusted = apply_adjustments(panel.drop(
            [c for c in panel.columns if c.startswith("adj_") or c == "divisor"]
        ), agreed)
        write_derived_parquet(adjusted, cfg.paths.data_processed / "prices_adjusted.parquet")

        if disputed:
            write_derived_parquet(
                pl.DataFrame({
                    "symbol": [d.symbol for d in disputed],
                    "ex_date": [d.ex_date for d in disputed],
                    "declared_factor": [d.declared_factor for d in disputed],
                    "implied_factor": [d.implied_factor for d in disputed],
                }),
                cfg.paths.data_processed / "disputed_corporate_actions.parquet",
            )

        run.note("symbols_queried", len(symbols))
        run.note("declared_events", len(declared))
        run.note("applied_events", len(agreed))
        run.note("disputed_events", len(disputed))
        run.note("symbol_fetch_failures", failed)

    _log.info("declared=%d applied=%d disputed=%d failed_symbols=%d",
              len(declared), len(agreed), len(disputed), len(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
