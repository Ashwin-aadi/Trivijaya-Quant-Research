"""Record how much capital each P4 strategy actually deploys, per arm.

**Why this exists.** Phase 4.2's capacity run produced binding capacities of 727, 1283 and 372 crore
in G2, G6 and G7, against 10-37 crore everywhere else. They are not floating-point residue -- the
``min_traded_fraction`` guard added at Checkpoint 3.3 is 1e-9 and these strategies turn over 1e-5 to
1e-4 per session, so they are genuinely trading. They are strategies that hold **almost no book**:
``G2/candidate_435`` holds 100 names summing to 0.28% of equity, ``G6/candidate_144`` 60 names
summing to 1.5%, against a fully-invested 1.0 for a normal strategy.

Capacity is defined per rupee of AUM, not per rupee deployed, so a strategy that leaves 99% of the
account in cash reports an enormous capacity that is arithmetically correct and substantively
empty. FlowState's five validation factors were all fully invested, so the validation set could not
reach this path -- the same shape of gap the Phase 3.3 corpus run found, found the same way.

**Nothing is fixed here.** FlowState is frozen and the Phase 3.3 ruling is explicit that a corpus
finding is to be reported, never used to adjust the benchmark. This script only measures how common
the pattern is, so the capacity figures can be reported against the exposure they were earned at.

Writes ``benchmarks/generationbench/corpus/<arm>/exposure.json``. One arm per write.

Usage:
    python scripts/paradigm_exposure.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import numpy as np  # noqa: E402
from paradigm_capacity import traded_paths  # noqa: E402
from run_corpus_backtest import _instantiate, _load_strategy  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.common.config import Config, load_config  # noqa: E402
from src.common.io import read_parquet  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
# Below this median gross exposure a strategy is holding essentially cash, and its capacity figure
# describes an account that is not being invested. Stated here as a reporting threshold, not as a
# filter: nothing is dropped on it, every strategy is written out with its measured exposure.
NEAR_CASH = 0.10


def exposure_arm(arm: str, engine: BacktestEngine, cfg: Config) -> dict[str, Any]:
    """Median and maximum gross exposure, and names held, for every traded strategy in `arm`."""
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for name, path in traded_paths(arm):
        result = engine.run(_instantiate(_load_strategy(path)),
                            start=cfg.dates.dev_start, end=cfg.dates.dev_end)
        gross = np.asarray(result.gross_exposure, dtype=float)
        held = [len(book) for book in result.positions]
        if gross.size == 0:
            continue
        records.append({
            "name": name,
            "median_gross_exposure": float(np.median(gross)),
            "max_gross_exposure": float(gross.max()),
            "median_names_held": int(np.median(held)) if held else 0,
            "near_cash": bool(np.median(gross) < NEAR_CASH),
        })

    near = [r for r in records if r["near_cash"]]
    _log.info("%-3s %3d strategies in %.1f min; %d near-cash (<%.0f%% deployed), "
              "median gross exposure %.3f", arm, len(records),
              (time.perf_counter() - started) / 60, len(near), NEAR_CASH * 100,
              float(np.median([r["median_gross_exposure"] for r in records])) if records else 0.0)

    return {
        "arm": arm, "paradigm": ARMS[arm],
        "n_strategies": len(records),
        "near_cash_threshold": NEAR_CASH,
        "n_near_cash": len(near),
        "exposure": records,
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    args = parser.parse_args()

    cfg = load_config()
    arms = [args.arm] if args.arm else list(ARMS)
    total = sum(len(traded_paths(a)) for a in arms)
    _log.info("%d traded strategies across %s; estimate %.0f min",
              total, ",".join(arms), total * 9 / 60)

    # The unfiltered panel, for the reason recorded in paradigm_capacity.main.
    engine = BacktestEngine(
        panel=read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet"),
        calendar=load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet"),
        universe=read_parquet(cfg.paths.data_processed / "universe.parquet"),
        cost_model=CostModel(cfg.costs),
        record_positions=True,
    )

    for arm in arms:
        payload = exposure_arm(arm, engine, cfg)
        out = CORPUS / arm / "exposure.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
