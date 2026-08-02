"""Store each strategy's per-session target book from the real panel — the Phase 2.2 feature input.

Holding period and concentration are properties of *positions*, and positions cannot be recovered
from an equity curve: a strategy that buys ten names and holds them for a quarter and one that
churns one name daily can produce the same return series. Until now the engine discarded its book
each session because nothing needed it. The PI approved recording it on 2026-08-02 as
instrumentation rather than a methodological change, and
``tests/backtest/test_engine.py::test_recording_positions_changes_no_number_the_engine_reports``
asserts that turning it on leaves every reported series identical.

This is one backtest per strategy on the **real, unmodified** development panel — the same run
``persist_real_returns.py`` already did, with recording switched on. It is not a re-run of the
Tier 1 suite and touches no synthetic path.

Each worker writes its own ``data/interim/positions/<name>.parquet`` rather than shipping roughly
ten million rows back through the process pool. Knife-edge strategies are stored and flagged, not
skipped, so the stability analysis can describe them.

Usage:
    python scripts/persist_positions.py --workers 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import polars as pl  # noqa: E402
from persist_real_returns import knife_edge_names  # noqa: E402
from run_corpus_backtest import _instantiate  # noqa: E402
from run_stress_tier1 import _load_all, load_dev_panel, strategy_paths  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402

_log = get_logger(__name__)

OUT_DIR = Path("data/interim/positions")

_STATE: dict[str, Any] = {}


def _initialise() -> None:
    """One engine per worker, built over the real panel with position recording enabled."""
    cfg = load_config()
    panel, universe = load_dev_panel(cfg)
    _STATE.update(
        cfg=cfg,
        engine=BacktestEngine(
            panel=panel,
            calendar=load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet"),
            universe=universe,
            cost_model=CostModel(cfg.costs),
            record_positions=True,
        ),
    )


def _one(name: str, source: str) -> dict[str, Any]:
    """Run one strategy and write its book to parquet. Never raises: a failure is a datum."""
    if "classes" not in _STATE:
        _STATE["classes"] = {}
    if name not in _STATE["classes"]:
        _STATE["classes"].update(_load_all([(name, source)]))
    cls = _STATE["classes"][name]
    if isinstance(cls, str):
        return {"name": name, "outcome": "import_error", "error": cls}
    cfg = _STATE["cfg"]
    try:
        result = _STATE["engine"].run(
            _instantiate(cls), start=cfg.dates.dev_start, end=cfg.dates.dev_end
        )
    except Exception as exc:  # noqa: BLE001 - untrusted strategy code
        return {"name": name, "outcome": "runtime_error",
                "error": f"{type(exc).__name__}: {exc}"[:200]}

    dates: list[Any] = []
    symbols: list[str] = []
    weights: list[float] = []
    for day, book in zip(result.dates, result.positions, strict=True):
        for symbol, weight in book.items():
            dates.append(day)
            symbols.append(symbol)
            weights.append(float(weight))

    # A cash-only strategy holds nothing on every session and legitimately produces no rows. Its
    # session count still has to survive, or the feature stage would silently see a shorter series,
    # so the sessions are carried in the return value rather than inferred from the file.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"session_date": dates, "symbol": symbols, "weight": weights},
        schema={"session_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64},
    ).write_parquet(OUT_DIR / f"{name}.parquet")

    return {
        "name": name,
        "outcome": "evaluated",
        "n_sessions": len(result.dates),
        "n_rows": len(dates),
        "first_session": result.dates[0].isoformat(),
        "last_session": result.dates[-1].isoformat(),
        "mean_turnover": float(sum(result.turnover) / len(result.turnover)),
        # The turnover *series*, not just its mean: turnover volatility distinguishes a strategy
        # that trades steadily from one that sits still and then rebuilds its whole book, and those
        # are plausibly different kinds of fragile. 192k floats in total, so it travels back through
        # the pool rather than becoming a third file.
        "turnover": [float(x) for x in result.turnover],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--strategies", type=int, default=None)
    args = parser.parse_args()

    # Same pinning as every other pooled run here: separate processes, so an unfixed hash seed
    # would make even a single deterministic pass irreproducible.
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["POLARS_MAX_THREADS"] = "1"

    cfg = load_config()
    entries = strategy_paths(args.strategies)
    knife = knife_edge_names()
    _log.info(
        "recording books for %d strategies (%d flagged knife-edge), %d workers",
        len(entries), sum(1 for n, _ in entries if n in knife), args.workers,
    )

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialise) as pool:
        futures = {pool.submit(_one, name, source): name for name, source in entries}
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 20 == 0:
                _log.info("%d/%d complete (%.1f min)", done, len(entries),
                          (time.perf_counter() - started) / 60)
    wall = time.perf_counter() - started

    evaluated = [r for r in results if r["outcome"] == "evaluated"]
    failures = [(r["name"], r.get("error")) for r in results if r["outcome"] != "evaluated"]

    with RunManifest(cfg, script="persist_positions.py") as run:
        turnover = pl.concat([
            pl.DataFrame({
                "name": r["name"],
                "session_index": list(range(r["n_sessions"])),
                "turnover": r["turnover"],
            })
            for r in evaluated
        ])
        turnover.write_parquet(cfg.paths.data_processed / "session_turnover.parquet")

        index = pl.DataFrame([
            {k: v for k, v in r.items() if k != "turnover"} for r in evaluated
        ]).with_columns(
            knife_edge=pl.col("name").is_in(sorted(knife))
        ).sort("name")
        out = cfg.paths.data_processed / "position_index.parquet"
        index.write_parquet(out)
        run.note("strategies_stored", index.height)
        run.note("strategies_failed", len(failures))
        run.note("position_rows", int(index["n_rows"].sum()))

    print(f"\n  strategies requested   {len(entries)}")
    print(f"  stored                 {len(evaluated)}")
    print(f"  failed                 {len(failures)}")
    for name, error in failures:
        print(f"      {name:24s} {error}")
    print(f"  position rows          {int(index['n_rows'].sum()):,}")
    print(f"  sessions per strategy  {index['n_sessions'].min()} min / "
          f"{index['n_sessions'].max()} max")
    print(f"  wall clock             {wall / 60:.1f} min")
    print(f"  books written to       {OUT_DIR}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
