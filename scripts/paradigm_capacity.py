"""Apply the frozen FlowState benchmark to each P4 arm, changing only the population.

Follows ``run_frontier_capacity.py``, which did this for the frontier arms, and obeys the same rule:
**this file contains no measurement code.** It departs from it in exactly one respect -- the price
panel handed to the engine, for the reason recorded at the load in :func:`main`. It records each
strategy's target book and hands it to :mod:`src.capacity.deployability` exactly as the frozen
benchmark defines it. If an arm behaves
unlike the reference corpus, that is a finding to report, never a reason to adjust anything.

**Every capacity figure here is a constraint figure, never an impact-erosion figure.** Phase 3.0
measured that daily bars cannot identify a transient impact function, so the two answer different
questions and only one of them is supported by the data. Nothing in this file may blur them.

**Why the artifacts are namespaced.** ``run_corpus_capacity.py`` reads
``data/processed/position_index.parquet`` and P2's stress run writes ``data/interim/positions/``,
both frozen P3 artifacts describing the 156-strategy reference corpus. P4's candidates are also
named ``candidate_000`` upward, so writing there would silently replace files belonging to a
released benchmark -- a parquet write does not warn about what it overwrites. Everything here stays
under ``benchmarks/generationbench/corpus/<arm>/``.

Only strategies that took a position are run. A candidate holding no book has no turnover for a
participation limit to bind against.

One arm per write, so an interrupted run loses at most the arm in flight.

Usage:
    python scripts/paradigm_capacity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import polars as pl  # noqa: E402
from run_corpus_backtest import _instantiate, _load_strategy  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.capacity.deployability import (  # noqa: E402
    capacity_by_flow_state,
    session_capacity,
    summarise_capacity,
    turnover_by_session,
)
from src.capacity.impact import add_daily_measures  # noqa: E402
from src.common.config import Config, load_config  # noqa: E402
from src.common.io import read_parquet  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")


def traded_paths(arm: str) -> list[tuple[str, Path]]:
    """(name, source) for each candidate in `arm` that executed and took a position."""
    results = json.loads((CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
    return [
        (row["name"], Path(row["path"]))
        for row in results
        if row["outcome"] == "evaluated" and (row.get("mean_turnover") or 0) > 0
    ]


def record_books(arm: str, engine: BacktestEngine, cfg: Config) -> pl.DataFrame:
    """Every traded strategy's target book per session, stacked into one frame.

    ``deployability`` names its grouping column ``factor`` because Phase 3.1 fed it factor books.
    It is a strategy identifier, not a claim that these are factors, and it is reused rather than
    renamed so that not one line of the frozen module changes for this run.
    """
    frames: list[pl.DataFrame] = []
    for name, path in traded_paths(arm):
        strategy = _instantiate(_load_strategy(path))
        result = engine.run(strategy, start=cfg.dates.dev_start, end=cfg.dates.dev_end)
        rows = [
            {"session_date": day, "factor": name, "symbol": sym, "weight": float(w)}
            for day, book in zip(result.dates, result.positions, strict=True)
            for sym, w in book.items()
        ]
        if rows:
            frames.append(pl.DataFrame(rows))
    return pl.concat(frames) if frames else pl.DataFrame()


def capacity_arm(arm: str, engine: BacktestEngine, cfg: Config,
                 liquidity: pl.DataFrame, flows: pl.DataFrame) -> dict[str, Any]:
    """Constraint-based deployment capacity for one arm, by the frozen FlowState definition."""
    started = time.perf_counter()
    positions = record_books(arm, engine, cfg)
    if positions.is_empty():
        _log.warning("%s: no position rows recorded", arm)
        return {"arm": arm, "n_strategies": 0, "capacity": []}

    limit = cfg.constraints.max_participation_rate
    traded = turnover_by_session(
        positions, min_traded_fraction=cfg.constraints.min_traded_fraction)
    per_session = session_capacity(traded, liquidity, participation_limit=limit)
    summaries = summarise_capacity(per_session, participation_limit=limit)

    crores = sorted(s.binding_capacity_inr / 1e7 for s in summaries)
    _log.info("%-3s %3d strategies, %d position rows in %.1f min; binding crore "
              "min %.2f median %.2f max %.2f", arm, positions["factor"].n_unique(),
              positions.height, (time.perf_counter() - started) / 60,
              crores[0], crores[len(crores) // 2], crores[-1])

    return {
        "arm": arm, "paradigm": ARMS[arm],
        "participation_limit": limit,
        "min_traded_fraction": cfg.constraints.min_traded_fraction,
        "n_strategies": positions["factor"].n_unique(),
        "measure": "constraint-based deployment capacity, never impact erosion",
        "capacity": [asdict(s) for s in summaries],
        # A frame, not a mapping: it carries the session count per flow state alongside the median,
        # and the frozen module's docstring is explicit that the ratio must never be quoted without
        # both sample sizes.
        "by_flow_state": capacity_by_flow_state(per_session, flows).to_dicts(),
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    args = parser.parse_args()

    cfg = load_config()
    arms = [args.arm] if args.arm else list(ARMS)
    total = sum(len(traded_paths(a)) for a in arms)
    # ~9 s per strategy, serial, measured in runs/frontier_gpt_capacity.log (20 in 184.7 s).
    _log.info("%d traded strategies across %s; estimate %.0f min",
              total, ",".join(arms), total * 9 / 60)

    # The **unfiltered** panel, matching ``run_corpus_backtest._worker_init`` exactly. This is not a
    # cosmetic choice. ``prices_adjusted.parquet`` begins on 2019-06-25 while trading begins on
    # ``dev_start`` (2020-01-01), and those 188,853 pre-window rows are the warm-up a lookback
    # strategy needs on its first trading session. ``load_dev_panel`` discards them, so a 60-session
    # strategy run through it sees nothing on day one -- G2/candidate_045 divides by the count of
    # symbols with a full window and raised ZeroDivisionError there while completing 1232 sessions
    # here. Every P4 Sharpe, turnover and return series came from the unfiltered panel, so capacity
    # must too, or a strategy's capacity figure would describe a different strategy from the one in
    # the same row. PI ruling of 2026-08-08.
    #
    # The liquidity frame below is *deliberately* still dev-window-filtered: it mirrors
    # ``run_corpus_capacity.py`` line for line, so the capacity definition stays P3's.
    panel = read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = read_parquet(cfg.paths.data_processed / "universe.parquet")
    engine = BacktestEngine(
        panel=panel,
        calendar=load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet"),
        universe=universe,
        cost_model=CostModel(cfg.costs),
        record_positions=True,
    )
    prices = read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet").filter(
        (pl.col("session_date") >= cfg.dates.dev_start)
        & (pl.col("session_date") <= cfg.dates.dev_end)
    )
    liquidity = add_daily_measures(
        prices, adv_window=cfg.constraints.adv_window_sessions
    ).select(["session_date", "symbol", "adv_inr"])
    flows = read_parquet(cfg.paths.data_processed / "participant_flows.parquet")

    for arm in arms:
        payload = capacity_arm(arm, engine, cfg, liquidity, flows)
        out = CORPUS / arm / "capacity.json"
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
