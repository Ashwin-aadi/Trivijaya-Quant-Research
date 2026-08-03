"""Apply P2's frozen duplicate criterion to one frontier arm's realised return series.

Redundancy is the metric a generator study is most likely to get wrong by writing its own version
of it. A model that finds one workable idea and emits it repeatedly looks strong on any average and
is worth nothing, and whether that happens is precisely what the addendum's H4 asks. So the
criterion is not re-derived here: ``_clusters``, ``_near_pairs`` and both thresholds are imported
from :mod:`deduplicate_corpus`, the script P2 released. If that criterion is ever revised, this
script changes with it and the two populations stay comparable.

**Duplication is judged on realised net returns, not on source text.** Two strategies with different
class names, different parameters and different code can trade identically, and several in the local
corpus did. Comparing sources would report them as distinct and overstate the diversity of every
arm.

Usage:
    python scripts/frontier_duplicates.py --arm claude
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

from deduplicate_corpus import (  # noqa: E402
    EXACT_TOLERANCE,
    NEAR_CORRELATION,
    _clusters,
    _near_pairs,
)

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def load_returns(arm: str) -> tuple[list[str], np.ndarray, list[str]]:
    """``(names, matrix, uncompared)`` for the arm, matrix rows aligned to names.

    The backtests were written before the arm was renamed away from ``candidate_NNN``, so the files
    still carry that prefix. The pooling record maps position to the arm's name and is used to
    relabel, rather than trusting two independent sorts to agree.
    """
    run = ROOT / "runs" / f"frontier_{arm}"
    pooled = json.loads((run / "pooling.json").read_text(encoding="utf-8"))
    labels = [str(entry["candidate"]).removesuffix(".py") for entry in pooled["index"]]

    frames: dict[str, np.ndarray] = {}
    for position, label in enumerate(labels):
        path = run / "backtests_development" / f"candidate_{position:03d}_returns.parquet"
        if not path.exists():
            continue
        frames[label] = pl.read_parquet(path)["return"].to_numpy()

    # Only equal-length series are comparable elementwise, exactly as P2 requires. A strategy that
    # stopped early is reported as uncompared rather than dropped from the census.
    full = max(len(v) for v in frames.values())
    names = sorted(n for n, v in frames.items() if len(v) == full)
    uncompared = sorted(n for n, v in frames.items() if len(v) != full)
    matrix = np.column_stack([frames[n] for n in names]).T
    return names, matrix, uncompared


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    names, matrix, uncompared = load_returns(args.arm)
    clusters = _clusters(names, matrix)
    duplicated = {name for cluster in clusters for name in cluster}
    near = _near_pairs(names, matrix, duplicated)

    payload = {
        "arm": args.arm,
        "criterion": "P2 frozen: identical net return on every session to within EXACT_TOLERANCE",
        "exact_tolerance": EXACT_TOLERANCE,
        "near_correlation_threshold": NEAR_CORRELATION,
        "n_compared": len(names),
        "n_uncompared": len(uncompared),
        "uncompared": uncompared,
        "n_exact_clusters": len(clusters),
        "n_in_a_cluster": len(duplicated),
        "clusters": clusters,
        "near_duplicate_pairs": near,
    }
    out = ROOT / "runs" / f"frontier_{args.arm}" / "duplicates.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _log.info("arm %s: %d compared, %d exact clusters covering %d strategies",
              args.arm, len(names), len(clusters), len(duplicated))
    for cluster in clusters:
        _log.info("  cluster: %s", ", ".join(cluster))
    _log.info("  near-duplicate pairs above %.4f not already merged: %d",
              NEAR_CORRELATION, len(near))
    return 0


if __name__ == "__main__":
    sys.exit(main())
