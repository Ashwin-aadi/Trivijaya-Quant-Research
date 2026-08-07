"""P2's Tier 1 calibration over each P4 arm: nondeterminism and reconstruction sensitivity.

Tier 2 excludes knife-edge strategies from fragility under the PI's ruling of 2026-08-02, so the P4
corpus cannot enter the stress stage until it has been classified the same way P1's and the frontier
arms' corpora were. That makes this a prerequisite rather than an extra.

It is also a result in its own right. P2 reported 19.9% of the local corpus sitting on a knife edge
and 17.3% nondeterministic, as findings about machine-written strategies; the generator-validation
study then measured both across frontier models. Neither varied the *methodology*, so whether
scaffolding reduces these pathologies is open, and the measurement falls out of a step that has to
happen anyway.

**No measurement is redefined here.** ``run_group``, ``compare`` and the tolerance are imported from
:mod:`calibrate_tier1` unchanged, exactly as ``frontier_calibrate.py`` does; this script supplies
only the list of strategies.

The three comparisons, following P2's design:

* **A vs A2** -- identical panel, identical hash seed. Movement is unseeded randomness inside the
  strategy itself, and is subtracted before the other two are read.
* **A vs B** -- identical panel, two hash seeds. Movement means the strategy iterates an unordered
  container, so its portfolio depends on hash order.
* **A vs C** -- real panel against its identity reconstruction. Movement means the strategy
  amplifies a 9e-15 residual into a visible Sharpe difference: a knife edge.

Only position-taking strategies are calibrated. A candidate that never traded has no Sharpe for a
panel difference to move, so including it would dilute every rate with strategies that could not
have exhibited the property.

Writes ``benchmarks/generationbench/corpus/<arm>/calibration.json``. One arm per write, so an
interrupted run loses at most the arm in flight.

Usage:
    python scripts/paradigm_calibrate.py --workers 8
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

from calibrate_tier1 import TOLERANCE, compare, run_group  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")


def arm_entries(arm: str) -> list[tuple[str, str]]:
    """(name, source path) for every candidate in `arm` that executed and took a position."""
    results = json.loads((CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
    return [
        (row["name"], row["path"])
        for row in results
        if row["outcome"] == "evaluated" and (row.get("mean_turnover") or 0) > 0
    ]


def _rows(moved: list[tuple[str, float, float, float]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "left": left, "right": right, "abs_diff": diff}
        for name, left, right, diff in moved
    ]


def calibrate_arm(arm: str, workers: int) -> dict[str, Any]:
    """Classify one arm's traded strategies as unseeded-random, nondeterministic, or knife-edge."""
    entries = arm_entries(arm)
    if not entries:
        return {"arm": arm, "n_strategies": 0}

    started = time.perf_counter()
    group_a = run_group(entries, identity=False, hash_seed=0, workers=workers)
    group_a2 = run_group(entries, identity=False, hash_seed=0, workers=workers)
    group_b = run_group(entries, identity=False, hash_seed=1, workers=workers)
    group_c = run_group(entries, identity=True, hash_seed=0, workers=workers)
    wall = (time.perf_counter() - started) / 60

    unseeded = compare(group_a, group_a2)
    nondeterministic = compare(group_a, group_b)
    reconstruction = compare(group_a, group_c)

    # A strategy already shown to be noisy at a fixed seed cannot also be shown to be
    # reconstruction-sensitive: its own noise explains the difference. P2 subtracts it, and so does
    # this, so every corpus in the programme is classified by the same rule.
    noisy = {row[0] for row in unseeded} | {row[0] for row in nondeterministic}
    knife_edge = [row for row in reconstruction if row[0] not in noisy]

    _log.info("%-3s %3d traded in %.1f min: unseeded %d, nondet %d, knife-edge %d (%.1f%%)",
              arm, len(entries), wall, len(unseeded), len(nondeterministic), len(knife_edge),
              100.0 * len(knife_edge) / len(entries))

    return {
        "arm": arm, "paradigm": ARMS[arm], "n_strategies": len(entries),
        "tolerance": TOLERANCE, "wall_minutes": wall,
        "criterion_nondeterministic": (
            "net Sharpe changes across a replicate at fixed PYTHONHASHSEED, or across two hash "
            "seeds; tolerance 1e-9"
        ),
        "criterion_knife_edge": (
            "deterministic by the above, yet net Sharpe differs by more than the tolerance between "
            "the real panel and its identity reconstruction"
        ),
        "n_unseeded_random": len(unseeded), "unseeded_random": _rows(unseeded),
        "n_nondeterministic": len(nondeterministic), "nondeterministic": _rows(nondeterministic),
        "n_reconstruction_sensitive": len(reconstruction),
        "n_knife_edge": len(knife_edge), "knife_edge": _rows(knife_edge),
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    parser.add_argument("--workers", type=int, default=8,
                        help="8 by default: the semantic layer holds the GPU and RAM is 15.6 GB")
    args = parser.parse_args()

    arms = [args.arm] if args.arm else list(ARMS)
    total = sum(len(arm_entries(a)) for a in arms)
    # Four backtest groups per strategy, ~7.3 s each, ~5.8x effective parallelism measured in
    # runs/frontier_gpt_stress.log. Reported before the run per RULE 5.
    _log.info("%d traded strategies across %s; 4 groups each, estimate %.0f min",
              total, ",".join(arms), total * 4 * 7.3 / 5.8 / 60)

    for arm in arms:
        payload = calibrate_arm(arm, args.workers)
        out = CORPUS / arm / "calibration.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
