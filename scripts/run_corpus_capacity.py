"""Apply the frozen FlowState benchmark to the machine-generated strategy corpus.

Phase 3.3, ordered by the PI on 2026-08-03 and moved ahead of the write-up on the same day. This is
what gives every surviving local-model strategy a result from all three benchmarks — an audit
verdict from P1, a fragility score from P2, and a deployment capacity from P3 — and it is the
reference corpus the redefined addendum will compare frontier generators against.

**The instrument is frozen and this script may not change it.** FlowState was built and validated on
standard factor strategies in Phase 3.1 and its parameters were fixed before this corpus was ever
passed through it. That ordering is the whole point: a benchmark tuned against the corpus it will
later judge is not a benchmark. Accordingly this file contains **no measurement code**. It reads
positions, calls :mod:`src.capacity.deployability` unchanged, and joins the result to the two
existing benchmark artifacts. If the corpus behaves unlike the factor strategies, that is a finding
to report at Checkpoint 3.3 — not a reason to adjust anything.

Writes ``data/processed/corpus_capacity.json`` and ``corpus_capacity.parquet``.

Usage:
    python scripts/run_corpus_capacity.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from src.capacity.deployability import (
    capacity_by_flow_state,
    session_capacity,
    summarise_capacity,
    turnover_by_session,
)
from src.capacity.impact import add_daily_measures
from src.common.config import Config, load_config
from src.common.exceptions import DataIntegrityError
from src.common.io import read_parquet, write_derived_parquet
from src.common.log import get_logger
from src.common.manifest import RunManifest

_log = get_logger(__name__)


def _load_corpus_positions(root: Path) -> pl.DataFrame:
    """Stack every persisted position book into one frame keyed by strategy name.

    ``deployability`` names its grouping column ``factor`` because Phase 3.1 fed it factor books.
    The column is a strategy identifier, not a claim that these are factors, and it is reused rather
    than renamed so that not one line of the frozen module changes for this run.
    """
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise DataIntegrityError(f"no position books under {root}")
    frames = [
        read_parquet(path).with_columns(factor=pl.lit(path.stem)).select(
            ["session_date", "factor", "symbol", "weight"]
        )
        for path in files
    ]
    return pl.concat(frames)


def _eligible(cfg: Config) -> tuple[set[str], dict[str, Any]]:
    """The deterministic survivors, and the provenance of every exclusion applied to get there."""
    excluded_path = Path("benchmarks/regimestress/excluded_nondeterministic.json")
    excluded = json.loads(excluded_path.read_text(encoding="utf-8"))
    nondeterministic = {row["name"] for row in excluded["excluded"]}

    index = read_parquet(cfg.paths.data_processed / "position_index.parquet")
    evaluated = set(index.filter(pl.col("outcome") == "evaluated")["name"].to_list())
    knife_edge = set(index.filter(pl.col("knife_edge"))["name"].to_list())

    return evaluated - nondeterministic, {
        "n_position_books": index.height,
        "n_evaluated": len(evaluated),
        "n_nondeterministic_excluded": len(nondeterministic & evaluated),
        "n_knife_edge_flagged": len(knife_edge),
        "exclusion_criterion": excluded["criterion"],
    }


def main() -> int:
    cfg = load_config()
    with RunManifest(cfg, "run_corpus_capacity") as run:
        run.add_input(cfg.paths.data_processed / "position_index.parquet")
        run.add_input(cfg.paths.data_processed / "prices_adjusted.parquet")

        eligible, provenance = _eligible(cfg)
        positions = _load_corpus_positions(cfg.paths.data_interim / "positions")
        corpus = positions.filter(pl.col("factor").is_in(list(eligible)))
        _log.info("corpus: %d strategies, %d position rows",
                  corpus["factor"].n_unique(), corpus.height)

        panel = read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet").filter(
            (pl.col("session_date") >= cfg.dates.dev_start)
            & (pl.col("session_date") <= cfg.dates.dev_end)
        )
        liquidity = add_daily_measures(
            panel, adv_window=cfg.constraints.adv_window_sessions
        ).select(["session_date", "symbol", "adv_inr"])

        limit = cfg.constraints.max_participation_rate
        traded = turnover_by_session(
            corpus, min_traded_fraction=cfg.constraints.min_traded_fraction
        )
        per_session = session_capacity(traded, liquidity, participation_limit=limit)
        summaries = summarise_capacity(per_session, participation_limit=limit)

        flows = read_parquet(cfg.paths.data_processed / "participant_flows.parquet")
        by_state = capacity_by_flow_state(per_session, flows)

        # Join to the other two benchmarks. Nothing is recomputed; these are read as frozen.
        fragility = json.loads(
            (cfg.paths.data_processed / "fragility.json").read_text(encoding="utf-8")
        )
        primary = fragility["primary"]
        if not isinstance(primary, list) or not primary:
            raise DataIntegrityError(
                "fragility.json['primary'] is not a non-empty list; the join key has moved and a "
                "silent zero-overlap count would look like a finding rather than a bug"
            )
        with_fragility = {row["name"] for row in primary}
        capacity_names = {s.factor for s in summaries}
        both = capacity_names & with_fragility
        if not both:
            raise DataIntegrityError(
                f"no strategy carries both a capacity and a fragility score "
                f"({len(capacity_names)} capacities, {len(with_fragility)} fragilities). The two "
                "artifacts do not share a naming convention, which is a defect, not a result."
            )

        write_derived_parquet(
            per_session, cfg.paths.data_processed / "corpus_capacity.parquet"
        )
        result = {
            "provenance": provenance,
            "n_eligible_deterministic_survivors": len(eligible),
            "n_with_capacity": len(capacity_names),
            "n_with_fragility": len(with_fragility),
            "n_with_all_three_benchmarks": len(both),
            "with_capacity_but_no_fragility": sorted(capacity_names - with_fragility),
            "participation_limit": limit,
            "capacity": [asdict(s) for s in summaries],
            "capacity_by_flow_state": by_state.to_dicts(),
        }
        out = cfg.paths.data_processed / "corpus_capacity.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        run.note("n_strategies", len(capacity_names))
        _log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
