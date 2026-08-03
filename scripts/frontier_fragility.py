"""Compute across-path fragility for one frontier arm from its completed stress run.

A thin wrapper, deliberately. ``src.stress.fragility.across_paths`` already takes a glob and already
handles the two cases that matter --- a strategy evaluated on only some paths, and a mean Sharpe so
close to zero that the variance-over-mean ratio explodes --- so the only thing an arm needs is its
own path pattern. Reimplementing the ratio here would create a second definition of the study's
fragility number, and the arms would stop being comparable the moment either drifted.

Usage:
    python scripts/frontier_fragility.py --arm claude
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402
from src.stress.fragility import across_paths  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    run = ROOT / "runs" / f"frontier_{args.arm}"
    pattern = str(run / "stress" / "path_*.json")
    fragility = across_paths(pattern)
    if not fragility:
        raise SystemExit(f"no evaluated stress results found under {pattern}")

    (run / "fragility.json").write_text(json.dumps(fragility, indent=1), encoding="utf-8")

    values = sorted(r["fragility_across_paths"] for r in fragility.values())
    near_zero = [n for n, r in fragility.items() if r["mean_is_near_zero"]]
    paths = {int(r["n_paths"]) for r in fragility.values()}
    _log.info("arm %s: %d strategies with across-path fragility", args.arm, len(fragility))
    _log.info("  paths per strategy: %s", sorted(paths))
    _log.info("  fragility: min %.3f  median %.3f  max %.3f",
              values[0], st.median(values), values[-1])
    # Near-zero mean Sharpe makes the ratio unstable rather than large. P2 reports these separately
    # rather than letting them dominate a median, so they are named here too.
    _log.info("  flagged mean-near-zero: %d%s",
              len(near_zero), (" (" + ", ".join(sorted(near_zero)) + ")") if near_zero else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
