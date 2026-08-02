"""Store each strategy's realised daily return series from the real panel — Tier 2's input.

Tier 2 bootstraps realised returns rather than rebuilding price history, so it needs one honest
return series per strategy to resample. Tier 1 did not leave one behind: its per-path files carry
summary metrics only, because persisting 100 paths x 158 strategies x 1,232 sessions of daily
returns would have written roughly 190 million floats to disk for no use at the time.

This is one backtest per strategy on the **real, unmodified** development panel — not a synthetic
path, not a re-run of the Tier 1 suite. Around three minutes at 24 workers against Tier 1's five
and a half hours.

The population is the same as Tier 1's, and for the same reason: a strategy whose returns are not
a stable function of its inputs would have those returns resampled here too, and the instability
would travel into every Tier 2 number. Knife-edge strategies are run and stored but **flagged**, so
the fragility stage can exclude them per the PI's ruling of 2026-08-02 while still reporting them.

Writes ``data/processed/real_returns.parquet``: one row per (strategy, session).

Usage:
    python scripts/persist_real_returns.py --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import polars as pl  # noqa: E402
from run_corpus_backtest import _instantiate  # noqa: E402
from run_stress_tier1 import _load_all, load_dev_panel, strategy_paths  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402

_log = get_logger(__name__)

KNIFE_EDGE = Path("benchmarks/regimestress/knife_edge.json")
OUT_NAME = "real_returns.parquet"

_STATE: dict[str, Any] = {}


def knife_edge_names() -> set[str]:
    """The frozen knife-edge set. Read, never recomputed, so the population cannot drift."""
    if not KNIFE_EDGE.exists():
        raise FileNotFoundError(f"{KNIFE_EDGE} is missing; run scripts/freeze_knife_edge.py first")
    payload = json.loads(KNIFE_EDGE.read_text(encoding="utf-8"))
    return {record["name"] for record in payload["knife_edge"]}


def _initialise() -> None:
    """One engine over the real panel per worker."""
    cfg = load_config()
    panel, universe = load_dev_panel(cfg)
    _STATE.update(
        cfg=cfg,
        engine=BacktestEngine(
            panel=panel,
            calendar=load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet"),
            universe=universe,
            cost_model=CostModel(cfg.costs),
        ),
    )


def _one(name: str, source: str) -> dict[str, Any]:
    """Run one strategy on the real panel and hand back its daily series. Never raises."""
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
    except Exception as exc:  # noqa: BLE001 - untrusted strategy code; a failure is a datum
        return {"name": name, "outcome": "runtime_error",
                "error": f"{type(exc).__name__}: {exc}"[:200]}
    return {
        "name": name,
        "outcome": "evaluated",
        "dates": [d.isoformat() for d in result.dates],
        "net": [float(x) for x in result.returns],
        "gross": [float(x) for x in result.gross_returns],
        "ruined_on": result.ruined_on.isoformat() if result.ruined_on else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--strategies", type=int, default=None)
    args = parser.parse_args()

    # Fixed for the same reason the Tier 1 runner fixes them: every worker is a separate process,
    # and an unfixed hash seed would make even this single-run step irreproducible.
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["POLARS_MAX_THREADS"] = "1"

    cfg = load_config()
    entries = strategy_paths(args.strategies)
    knife = knife_edge_names()
    _log.info(
        "persisting real return series for %d strategies (%d flagged knife-edge), %d workers",
        len(entries), sum(1 for n, _ in entries if n in knife), args.workers,
    )

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialise) as pool:
        futures = {pool.submit(_one, name, source): name for name, source in entries}
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - started

    with RunManifest(cfg, script="persist_real_returns.py") as run:
        frame = _to_frame(results, knife)
        out = cfg.paths.data_processed / OUT_NAME
        frame.write_parquet(out)
        failures = [(r["name"], r.get("error")) for r in results if r["outcome"] != "evaluated"]
        run.note("strategies_stored", frame["name"].n_unique())
        run.note("strategies_failed", len(failures))

    print(f"\n  strategies requested   {len(entries)}")
    print(f"  stored                 {frame['name'].n_unique()}")
    print(f"  failed                 {len(failures)}")
    for name, error in failures:
        print(f"      {name:20s} {error}")
    print(f"  sessions per strategy  {frame.group_by('name').len()['len'].min()}"
          f" min / {frame.group_by('name').len()['len'].max()} max")
    print(f"  knife-edge flagged     {frame.filter(pl.col('knife_edge'))['name'].n_unique()}")
    print(f"  wall clock             {wall / 60:.1f} min")
    print(f"  written to             {out}\n")
    return 0


def _to_frame(results: list[dict[str, Any]], knife: set[str]) -> pl.DataFrame:
    """Long format: one row per (strategy, session). Failures contribute no rows."""
    frames = [
        pl.DataFrame({
            "name": r["name"],
            "session_date": pl.Series(r["dates"]).str.to_date(),
            "net_return": r["net"],
            "gross_return": r["gross"],
        }).with_columns(knife_edge=pl.lit(r["name"] in knife))
        for r in results if r["outcome"] == "evaluated"
    ]
    if not frames:
        raise RuntimeError("no strategy produced a return series")
    return pl.concat(frames).sort("name", "session_date")


if __name__ == "__main__":
    sys.exit(main())
