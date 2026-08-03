"""Apply the frozen FlowState benchmark to one frontier arm, changing only the population.

Mirrors ``run_corpus_capacity.py``, which did this for the local corpus in Phase 3.3, and obeys the
same rule: **this file contains no measurement code.** It records each strategy's target book, then
hands it to :mod:`src.capacity.deployability` and :mod:`src.capacity.impact` exactly as the frozen
benchmark defines them. If the frontier arm behaves unlike the reference corpus, that is a finding
to report, never a reason to adjust anything.

**Why this is a separate script rather than a flag on the existing one.** ``run_corpus_capacity.py``
reads ``data/processed/position_index.parquet`` and ``run_stress_tier1.py`` writes
``data/interim/positions/``, both of which are frozen P3 artifacts describing the 156-strategy
reference corpus. The frontier arm's strategies were initially named ``candidate_000`` upward, which
collides with names already in that directory --- ``candidate_019.parquet`` exists there and belongs
to the local corpus. A parquet write does not warn about the file it replaces, so the collision
would have destroyed part of the frozen corpus without producing a single error message. Every
artifact here is therefore namespaced under ``runs/frontier_<arm>/``.

Usage:
    python scripts/run_frontier_capacity.py --arm gpt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import polars as pl  # noqa: E402
from run_corpus_backtest import _instantiate, _load_strategy  # noqa: E402
from run_stress_tier1 import load_dev_panel  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.capacity.deployability import (  # noqa: E402
    capacity_by_flow_state,
    session_capacity,
    summarise_capacity,
    turnover_by_session,
)
from src.capacity.impact import add_daily_measures  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.io import read_parquet  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def record_books(arm: str) -> pl.DataFrame:
    """Every strategy's target book per session, stacked into one frame.

    ``deployability`` names its grouping column ``factor`` because Phase 3.1 fed it factor books.
    It is a strategy identifier, not a claim that these are factors, and it is reused rather than
    renamed so that not one line of the frozen module changes for this run.
    """
    cfg = load_config()
    panel, universe = load_dev_panel(cfg)
    engine = BacktestEngine(
        panel=panel,
        calendar=load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet"),
        universe=universe,
        cost_model=CostModel(cfg.costs),
        record_positions=True,
    )
    corpus = ROOT / "runs" / f"frontier_{arm}" / "candidates"
    frames: list[pl.DataFrame] = []
    for path in sorted(p for p in corpus.glob("*.py") if p.stem != "__init__"):
        strategy = _instantiate(_load_strategy(path))
        result = engine.run(strategy, start=cfg.dates.dev_start, end=cfg.dates.dev_end)
        rows = [
            {"session_date": day, "factor": path.stem, "symbol": sym, "weight": float(w)}
            for day, book in zip(result.dates, result.positions, strict=True)
            for sym, w in book.items()
        ]
        if rows:
            frames.append(pl.DataFrame(rows))
        _log.info("  %s: %d sessions, %d position rows", path.stem, len(result.dates), len(rows))
    if not frames:
        raise SystemExit(f"no position rows recorded for arm {arm}")
    return pl.concat(frames)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    cfg = load_config()
    out = ROOT / "runs" / f"frontier_{args.arm}"
    with RunManifest(cfg, "run_frontier_capacity") as run:
        run.add_input(cfg.paths.data_processed / "prices_adjusted.parquet")
        run.note("arm", args.arm)

        positions = record_books(args.arm)
        positions.write_parquet(out / "positions.parquet")
        _log.info("recorded %d strategies, %d position rows",
                  positions["factor"].n_unique(), positions.height)

        panel = read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet").filter(
            (pl.col("session_date") >= cfg.dates.dev_start)
            & (pl.col("session_date") <= cfg.dates.dev_end)
        )
        liquidity = add_daily_measures(
            panel, adv_window=cfg.constraints.adv_window_sessions
        ).select(["session_date", "symbol", "adv_inr"])

        limit = cfg.constraints.max_participation_rate
        traded = turnover_by_session(
            positions, min_traded_fraction=cfg.constraints.min_traded_fraction
        )
        per_session = session_capacity(traded, liquidity, participation_limit=limit)
        summaries = summarise_capacity(per_session, participation_limit=limit)

        flows = read_parquet(cfg.paths.data_processed / "participant_flows.parquet")
        by_state = capacity_by_flow_state(per_session, flows)

        payload = {
            "arm": args.arm,
            "participation_limit": limit,
            "min_traded_fraction": cfg.constraints.min_traded_fraction,
            "n_strategies": positions["factor"].n_unique(),
            "capacity": [asdict(s) for s in summaries],
            # A frame, not a mapping: it carries the session count per flow state alongside the
            # median, and the frozen module's docstring is explicit that the ratio must never be
            # quoted without both sample sizes.
            "by_flow_state": by_state.to_dicts(),
        }
        (out / "capacity.json").write_text(json.dumps(payload, indent=2, default=str),
                                           encoding="utf-8")
        run.note("n_strategies", payload["n_strategies"])
        run.note("position_rows", positions.height)

    crores = sorted(s.binding_capacity_inr / 1e7 for s in summaries)
    _log.info("binding capacity, crore: min %.2f  median %.2f  max %.2f",
              crores[0], crores[len(crores) // 2], crores[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
