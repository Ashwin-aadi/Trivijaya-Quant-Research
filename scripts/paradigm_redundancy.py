"""Redundancy per generation paradigm, using P2's duplicate rule unchanged.

RQ3 predicts that heavier scaffolding buys quality by collapsing onto one idea and emitting it
repeatedly. That is only measurable if duplication is judged the same way it was judged in P2 --
otherwise a difference between the arms could be a difference between two detectors. So the
comparison and both thresholds are *imported* from `deduplicate_corpus.py` rather than restated
here; this script only chooses which series to feed them.

Duplication is judged on the realised net return series, never on source text: two strategies whose
code differs and whose returns agree on all 1,232 sessions are the same strategy for every purpose
downstream, and the converse is also true.

Only strategies that took a position are compared. A candidate that never traded has a constant
series and would merge with every other non-trader into one enormous cluster, which would say
nothing about the paradigm and would dominate the statistic.

Writes `benchmarks/generationbench/redundancy.json`.

Usage:
    python scripts/paradigm_redundancy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scripts.deduplicate_corpus import EXACT_TOLERANCE, NEAR_CORRELATION, _clusters  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
OUT = Path("benchmarks/generationbench/redundancy.json")


def _traded_series(arm: str) -> tuple[list[str], np.ndarray]:
    """Names and aligned net-return matrix for every candidate in `arm` that took a position."""
    results = json.loads((CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
    series: dict[str, np.ndarray] = {}
    for row in results:
        if row["outcome"] != "evaluated" or not (row.get("mean_turnover") or 0) > 0:
            continue
        path = row.get("returns_path")
        if not path or not Path(path).exists():
            continue
        series[row["name"]] = pl.read_parquet(path).sort("session_date")["return"].to_numpy()

    if not series:
        return [], np.empty((0, 0))

    # Elementwise comparison needs equal lengths. A strategy ruined early has a shorter series and
    # cannot be a duplicate of a complete one, so it is reported as uncompared, never dropped.
    full = max(len(v) for v in series.values())
    names = sorted(n for n, v in series.items() if len(v) == full)
    return names, np.vstack([series[n] for n in names])


def main() -> int:
    configure_logging()
    payload: dict[str, object] = {
        "purpose": (
            "Fraction of each paradigm's position-taking output inside an exact-duplicate cluster, "
            "judged on realised net returns by the identical rule and thresholds used in "
            "benchmarks/regimestress/duplicates.json."
        ),
        "exact_tolerance": EXACT_TOLERANCE,
        "near_correlation_threshold": NEAR_CORRELATION,
        "arms": {},
    }
    arms: dict[str, object] = payload["arms"]  # type: ignore[assignment]

    header = f"{'arm':4} {'traded':>6} {'compared':>8} {'clusters':>8} {'in cluster':>10} {'R':>7}"
    _log.info(header)
    for short in ARMS:
        names, matrix = _traded_series(short)
        n_results = len(json.loads(
            (CORPUS / short / "backtest_results.json").read_text(encoding="utf-8")))
        clusters = _clusters(names, matrix) if names else []
        duplicated = sorted({n for c in clusters for n in c})
        redundancy = len(duplicated) / len(names) if names else float("nan")

        arms[short] = {
            "paradigm": ARMS[short],
            "n_candidates": n_results,
            "n_traded": len(names),
            "n_clusters": len(clusters),
            "n_in_cluster": len(duplicated),
            "redundancy_of_traded": redundancy,
            "largest_cluster": max((len(c) for c in clusters), default=0),
            "clusters": clusters,
        }
        _log.info("%-4s %6d %8d %8d %10d %6.1f%%", short, len(names), len(names),
                  len(clusters), len(duplicated), redundancy * 100)

    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
