"""Run P2's Tier 1 calibration over one frontier arm: nondeterminism and reconstruction sensitivity.

Two properties of the local corpus were reported by RegimeStress as findings about machine-written
strategies --- that some are nondeterministic across hash seeds, and that a fifth sit on a knife
edge where a 9e-15 panel difference moves their Sharpe visibly. Both were measured on one local
model's output, so both are claims the generator-validation study should test rather than assume.

**No measurement is redefined here.** ``run_group``, ``compare`` and the tolerance are imported from
:mod:`calibrate_tier1` unchanged; this script supplies only the list of strategies, exactly as
``run_frontier_stress.py`` does for the stress suite.

The three comparisons, following P2's design:

* **A vs A2** --- identical panel, identical hash seed. Any movement is unseeded randomness inside
  the strategy itself, and must be subtracted before the other two can be read.
* **A vs B** --- identical panel, two different hash seeds. Movement means the strategy iterates an
  unordered container, so its portfolio depends on hash order.
* **A vs C** --- real panel against its identity reconstruction, same hash seed. Movement means the
  strategy amplifies a 9e-15 residual into a visible Sharpe difference: a knife edge.

Usage:
    python scripts/frontier_calibrate.py --arm gpt --workers 24
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
from run_frontier_stress import arm_entries  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def _rows(moved: list[tuple[str, float, float, float]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "left": left, "right": right, "abs_diff": diff}
        for name, left, right, diff in moved
    ]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config()
    entries = arm_entries(args.arm)
    _log.info("arm %s: calibrating %d strategies on %d workers, tolerance %g",
              args.arm, len(entries), args.workers, TOLERANCE)

    with RunManifest(cfg, script="frontier_calibrate.py") as run:
        started = time.perf_counter()
        group_a = run_group(entries, identity=False, hash_seed=0, workers=args.workers)
        group_a2 = run_group(entries, identity=False, hash_seed=0, workers=args.workers)
        group_b = run_group(entries, identity=False, hash_seed=1, workers=args.workers)
        group_c = run_group(entries, identity=True, hash_seed=0, workers=args.workers)
        wall = (time.perf_counter() - started) / 60

        unseeded = compare(group_a, group_a2)
        nondeterministic = compare(group_a, group_b)
        reconstruction = compare(group_a, group_c)

        # A strategy already shown to be noisy at a fixed seed cannot also be shown to be
        # reconstruction-sensitive: its own noise explains the difference. P2 subtracts it, and so
        # does this, so the two populations are classified by the same rule.
        noisy = {row[0] for row in unseeded} | {row[0] for row in nondeterministic}
        knife_edge = [row for row in reconstruction if row[0] not in noisy]

        payload = {
            "arm": args.arm,
            "n_strategies": len(entries),
            "tolerance": TOLERANCE,
            "wall_minutes": wall,
            "criterion_nondeterministic": (
                "net Sharpe changes across a replicate at fixed PYTHONHASHSEED, or across two "
                "hash seeds; tolerance 1e-9"
            ),
            "criterion_knife_edge": (
                "deterministic by the above, yet net Sharpe differs by more than the tolerance "
                "between the real panel and its identity reconstruction"
            ),
            "n_unseeded_random": len(unseeded),
            "unseeded_random": _rows(unseeded),
            "n_nondeterministic": len(nondeterministic),
            "nondeterministic": _rows(nondeterministic),
            "n_reconstruction_sensitive": len(reconstruction),
            "n_knife_edge": len(knife_edge),
            "knife_edge": _rows(knife_edge),
        }
        out = ROOT / "runs" / f"frontier_{args.arm}" / "calibration.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run.note("arm", args.arm)
        run.note("n_nondeterministic", len(nondeterministic))
        run.note("n_knife_edge", len(knife_edge))

    _log.info("arm %s in %.1f min: unseeded-random %d, nondeterministic %d, knife-edge %d of %d",
              args.arm, wall, len(unseeded), len(nondeterministic), len(knife_edge), len(entries))
    for name, left, right, diff in knife_edge:
        _log.warning("  knife edge: %s  %.6f vs %.6f  (%.3g)", name, left, right, diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
